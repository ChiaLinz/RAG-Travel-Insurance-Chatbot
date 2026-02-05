"""
Chunk Generator - 從 PDF 生成結構化條文

主要功能:
1. 解析 PDF 文本，識別章節、條文、項目、款項結構
2. 提取引用關係（如：第二十七條第一項第二款）
3. 生成階層式結構化 chunks
4. 保留完整的 context 和清理後的 raw_text
"""

import re
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from load_pdf import load_pdf
from config import INDEX_DIR


# ==================== 正則表達式模式 ====================

# 章節匹配：第一章 總則
CHAPTER_PATTERN = re.compile(r"^(第[一二三四五六七八九十百]+章)\s*(.+?)$", re.MULTILINE)

# 條文匹配：第一條 契約之構成
CLAUSE_PATTERN = re.compile(r"^(第[一二三四五六七八九十百]+條)\s*(.+?)\s*$", re.MULTILINE)

# 項目匹配：一、二、三、
ITEM_PATTERN = re.compile(r"^\s*([一二三四五六七八九十百]+)、", re.MULTILINE)

# 款項匹配：(一) 或 （1） 或 1) 或 (a)
SUBITEM_PATTERN = re.compile(
    r"^\s*(?:\(|（)([一二三四五六七八九十百]+|[0-9]+|[a-zA-Z])(?:\)|）)\s*",
    re.MULTILINE
)

# 引用關係匹配
# 匹配：前項、前款、第XX條、第XX條第X項、第XX條第X項第X款
REFERENCE_PATTERN = re.compile(
    r"(前項|前款|本條|"
    r"第[一二三四五六七八九十百]+條(?:第[一二三四五六七八九十百]+項)?(?:第[一二三四五六七八九十百]+款)?)",
    re.MULTILINE
)


# ==================== 數據結構 ====================

@dataclass
class SubItem:
    """款項結構"""
    subitem_no: str  # (一)、(1)、(a) 等
    context: str  # 完整文本（保留格式）
    raw_text: str  # 清理後文本（去除編號）
    reference_clauses: List[str]  # 引用的條文
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Item:
    """項目結構"""
    item_no: str  # 一、二、三
    context: str  # 完整文本
    raw_text: str  # 清理後文本
    sub_items: List[SubItem]  # 子款項
    reference_clauses: List[str]  # 引用的條文
    intent_ids: List[str]  # 關聯的意圖 ID（後續填充）
    
    def to_dict(self):
        return {
            "item_no": self.item_no,
            "context": self.context,
            "raw_text": self.raw_text,
            "sub_items": [si.to_dict() for si in self.sub_items],
            "reference_clauses": self.reference_clauses,
            "intent_ids": self.intent_ids
        }


@dataclass
class Clause:
    """條文主體結構"""
    clause_no: str  # 第一條
    clause_title: str  # 契約之構成
    clause_id: str  # 第一條_契約之構成
    context: str  # 完整條文內容
    raw_text: str  # 清理後內容
    items: List[Item]  # 子項目列表
    reference_clauses: List[str]  # 引用的條文
    intent_ids: List[str]  # 關聯的意圖 ID
    
    def to_dict(self):
        return {
            "clause_no": self.clause_no,
            "clause_title": self.clause_title,
            "clause_id": self.clause_id,
            "context": self.context,
            "raw_text": self.raw_text,
            "items": [item.to_dict() for item in self.items],
            "reference_clauses": self.reference_clauses,
            "intent_ids": self.intent_ids
        }


@dataclass
class Chunk:
    """完整的 chunk 結構（包含章節信息）"""
    chunk_id: str  # 與 clause_id 相同
    chapter_no: Optional[str]  # 第一章
    chapter_title: Optional[str]  # 總則
    clause: Clause  # 條文主體
    
    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "chapter_no": self.chapter_no,
            "chapter_title": self.chapter_title,
            "clause": self.clause.to_dict(),
            # 為了向後兼容，保留頂層字段
            "context": self.clause.context,
            "raw_text": self.clause.raw_text,
            "items": [item.to_dict() for item in self.clause.items],
            "intent_ids": self.clause.intent_ids,
            "reference_clauses": self.clause.reference_clauses
        }


# ==================== 輔助函數 ====================

def clean_text(text: str, remove_numbers: bool = True) -> str:
    """
    清理文本
    
    Args:
        text: 原始文本
        remove_numbers: 是否移除編號（一、(一)、1) 等）
    
    Returns:
        清理後的文本
    """
    # 移除多餘空白和換行
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_numbers:
        # 移除項目編號：一、二、三、
        text = re.sub(r'^[一二三四五六七八九十百]+、\s*', '', text)
        # 移除款項編號：(一)、（1）、1)、(a)
        text = re.sub(r'^(?:\(|（)[一二三四五六七八九十百0-9a-zA-Z]+(?:\)|）)\s*', '', text)
    
    return text


def detect_reference_clauses(text: str) -> List[str]:
    """
    檢測文本中的引用關係
    
    Returns:
        引用的條文列表，例如：['前項', '第二十七條', '第二十七條第一項第二款']
    """
    matches = REFERENCE_PATTERN.findall(text)
    # 去重並保持順序
    seen = set()
    result = []
    for ref in matches:
        if ref not in seen:
            seen.add(ref)
            result.append(ref)
    return result


def parse_subitems(text: str) -> List[SubItem]:
    """
    解析款項（第三層）
    
    Args:
        text: 包含款項的文本
    
    Returns:
        SubItem 列表
    """
    matches = list(SUBITEM_PATTERN.finditer(text))
    if not matches:
        return []
    
    subitems = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        subitem_text = text[start:end].strip()
        
        subitems.append(SubItem(
            subitem_no=match.group(1),
            context=subitem_text,
            raw_text=clean_text(subitem_text, remove_numbers=True),
            reference_clauses=detect_reference_clauses(subitem_text)
        ))
    
    return subitems


def parse_items(text: str) -> List[Item]:
    """
    解析項目（第二層）
    
    Args:
        text: 包含項目的文本
    
    Returns:
        Item 列表
    """
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return []
    
    items = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        item_text = text[start:end].strip()
        
        # 解析子款項
        sub_items = parse_subitems(item_text)
        
        items.append(Item(
            item_no=match.group(1),
            context=item_text,
            raw_text=clean_text(item_text, remove_numbers=True),
            sub_items=sub_items,
            reference_clauses=detect_reference_clauses(item_text),
            intent_ids=[]  # 稍後填充
        ))
    
    return items


def extract_chapter_info(text: str, clause_start: int) -> Tuple[Optional[str], Optional[str]]:
    """
    提取條文所屬的章節信息
    
    Args:
        text: 完整文本
        clause_start: 條文在文本中的起始位置
    
    Returns:
        (章節編號, 章節標題) 或 (None, None)
    """
    # 找到所有章節
    chapters = list(CHAPTER_PATTERN.finditer(text))
    
    # 找到最接近的前一個章節
    current_chapter_no = None
    current_chapter_title = None
    
    for chapter_match in chapters:
        if chapter_match.start() < clause_start:
            current_chapter_no = chapter_match.group(1)
            current_chapter_title = chapter_match.group(2).strip()
        else:
            break
    
    return current_chapter_no, current_chapter_title


# ==================== 主要函數 ====================

def generate_chunks_from_pdf(pdf_path: Optional[str] = None) -> List[Chunk]:
    """
    從 PDF 生成結構化 chunks
    
    Args:
        pdf_path: PDF 路徑（可選，使用 load_pdf 的默認路徑）
    
    Returns:
        Chunk 列表
    """
    # 載入 PDF 文本
    text = load_pdf().strip()
    
    # 找到所有條文
    clause_matches = list(CLAUSE_PATTERN.finditer(text))
    
    chunks = []
    
    for i, clause_match in enumerate(clause_matches):
        clause_no = clause_match.group(1)
        clause_title = clause_match.group(2).strip()
        clause_id = f"{clause_no}_{clause_title}"
        
        # 提取條文內容
        clause_start = clause_match.end()
        clause_end = clause_matches[i + 1].start() if i + 1 < len(clause_matches) else len(text)
        clause_body = text[clause_start:clause_end].strip()
        
        # 提取章節信息
        chapter_no, chapter_title = extract_chapter_info(text, clause_match.start())
        
        # 解析項目
        items = parse_items(clause_body)
        
        # 構建條文對象
        clause = Clause(
            clause_no=clause_no,
            clause_title=clause_title,
            clause_id=clause_id,
            context=clause_body,
            raw_text=clean_text(clause_body, remove_numbers=True),
            items=items,
            reference_clauses=detect_reference_clauses(clause_body),
            intent_ids=[]
        )
        
        # 構建 chunk
        chunk = Chunk(
            chunk_id=clause_id,
            chapter_no=chapter_no,
            chapter_title=chapter_title,
            clause=clause
        )
        
        chunks.append(chunk)
    
    return chunks


def save_chunks(chunks: List[Chunk], output_path: str):
    """
    保存 chunks 到 JSON 文件
    
    Args:
        chunks: Chunk 列表
        output_path: 輸出文件路徑
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": {
            "total_chunks": len(chunks),
            "generated_at": __import__('datetime').datetime.now().isoformat()
        },
        "chunks": [chunk.to_dict() for chunk in chunks]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已生成 {len(chunks)} 個 chunks，保存至 {output_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("🔄 開始生成 chunks...")
    
    # 生成 chunks
    chunks = generate_chunks_from_pdf()
    
    # 保存到文件
    output_path = os.path.join(INDEX_DIR, "chunks_structured.json")
    save_chunks(chunks, output_path)
    
    # 打印統計信息
    total_items = sum(len(chunk.clause.items) for chunk in chunks)
    total_subitems = sum(
        len(item.sub_items) 
        for chunk in chunks 
        for item in chunk.clause.items
    )
    
    print(f"📊 統計:")
    print(f"   - 總條文數: {len(chunks)}")
    print(f"   - 總項目數: {total_items}")
    print(f"   - 總款項數: {total_subitems}")
    
    # 顯示第一個 chunk 示例
    if chunks:
        print("\n📄 第一個 chunk 示例:")
        print(json.dumps(chunks[0].to_dict(), ensure_ascii=False, indent=2)[:500] + "...")
