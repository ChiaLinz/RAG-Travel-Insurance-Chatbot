"""
Chunk Generator V2 - 優化版條文解析器

主要優化：
1. 更精細的語義標註（區分「延誤」vs「遺失」vs「損失」）
2. 增強的引用關係解析（支援相對引用：前項、前款）
3. 關鍵詞提取（自動標註每個 chunk 的核心概念）
4. 條文分類（自動識別保險類型）
"""

import re
import json
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
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

# 引用關係匹配 - 增強版
REFERENCE_PATTERN = re.compile(
    r"(前項|前款|前條|本條|本項|本款|"
    r"第[一二三四五六七八九十百]+條(?:第[一二三四五六七八九十百]+項)?(?:第[一二三四五六七八九十百]+款)?)",
    re.MULTILINE
)


# ==================== 語義分類器 ====================

class SemanticClassifier:
    """語義分類器 - 自動識別條文類型和關鍵概念"""
    
    # 保險類型關鍵詞
    INSURANCE_TYPES = {
        "班機延誤": ["班機", "航班", "定期航班", "延誤", "預定出發時間"],
        "行李延誤": ["行李", "延誤", "抵達目的地", "未領得"],
        "行李損失": ["行李", "損失", "毀損", "滅失", "遺失", "竊盜", "強盜", "搶奪"],
        "旅程取消": ["旅程", "取消", "預定", "無法取回"],
        "旅程更改": ["旅程", "更改", "增加之交通", "住宿費用"],
        "租車事故": ["租用汽車", "租車", "駕駛", "交通事故"],
        "現金竊盜": ["現金", "竊盜", "強盜", "搶奪", "隨身攜帶"],
        "信用卡盜用": ["信用卡", "盜用", "掛失", "止付"],
        "急難救助": ["急難", "救助", "轉送", "搜索", "救援"],
    }
    
    # 條文功能類型
    CLAUSE_FUNCTIONS = {
        "承保範圍": ["承保範圍", "保險範圍", "本公司依本保險契約"],
        "不保事項": ["不保事項", "不負理賠責任", "除外責任", "特別不保"],
        "理賠文件": ["理賠文件", "申請理賠", "應檢具下列文件"],
        "理賠金額": ["保險金額", "給付", "理賠金額", "最高以"],
        "定義說明": ["所稱", "係指", "定義"],
    }
    
    # 關鍵動作詞（用於區分相似概念）
    ACTION_KEYWORDS = {
        "延誤": ["延誤", "延遲", "未於", "超過時間"],
        "遺失": ["遺失", "失蹤", "未尋獲"],
        "損失": ["損失", "毀損", "滅失"],
        "取消": ["取消", "終止", "中止"],
        "更改": ["更改", "變更", "調整"],
        "竊盜": ["竊盜", "偷竊"],
        "搶奪": ["搶奪", "搶劫", "強盜"],
    }
    
    @staticmethod
    def classify_insurance_type(text: str, clause_title: str) -> List[str]:
        """識別保險類型"""
        types = []
        combined_text = f"{clause_title} {text}"
        
        for insurance_type, keywords in SemanticClassifier.INSURANCE_TYPES.items():
            if any(kw in combined_text for kw in keywords):
                types.append(insurance_type)
        
        return types if types else ["其他"]
    
    @staticmethod
    def classify_clause_function(text: str, clause_title: str) -> str:
        """識別條文功能"""
        combined_text = f"{clause_title} {text}"
        
        for func_type, keywords in SemanticClassifier.CLAUSE_FUNCTIONS.items():
            if any(kw in combined_text for kw in keywords):
                return func_type
        
        return "一般規定"
    
    @staticmethod
    def extract_action_keywords(text: str) -> List[str]:
        """提取動作關鍵詞"""
        actions = []
        
        for action, keywords in SemanticClassifier.ACTION_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                actions.append(action)
        
        return actions
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """提取實體（時間、金額、數量等）"""
        entities = {
            "時間": [],
            "金額": [],
            "次數": [],
            "地點": []
        }
        
        # 時間實體
        time_patterns = [
            r"(\d+(?:小時|天|日|個月|年))",
            r"(二十四小時|四小時|六小時)",
            r"(預定出發時間|實際出發時間)"
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            entities["時間"].extend(matches)
        
        # 金額實體
        money_patterns = [
            r"(新臺幣\s*[零一二三四五六七八九十百千萬億\d]+元)",
            r"(保險金額)",
        ]
        for pattern in money_patterns:
            matches = re.findall(pattern, text)
            entities["金額"].extend(matches)
        
        # 次數實體
        count_patterns = [
            r"(給付[一二三四五六七八九十]+次)",
            r"([一二三四五六七八九十]+次事故)",
        ]
        for pattern in count_patterns:
            matches = re.findall(pattern, text)
            entities["次數"].extend(matches)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities


# ==================== 引用解析器 ====================

class ReferenceResolver:
    """引用解析器 - 處理相對和絕對引用"""
    
    def __init__(self, chunks: List[Dict]):
        """初始化解析器"""
        self.chunks = chunks
        self.clause_index = {}  # clause_no -> index
        
        # 建立索引
        for i, chunk in enumerate(chunks):
            clause_no = chunk["clause"]["clause_no"]
            self.clause_index[clause_no] = i
    
    def resolve_reference(self, 
                         ref: str, 
                         current_clause_no: str,
                         current_item_no: Optional[str] = None) -> str:
        """
        解析引用關係
        
        Args:
            ref: 引用文本（如："前項"、"第二十七條第一項"）
            current_clause_no: 當前條文編號
            current_item_no: 當前項目編號（可選）
        
        Returns:
            解析後的絕對引用（如："第二十六條第二項"）
        """
        # 絕對引用（已經是完整的）
        if ref.startswith("第") and "條" in ref:
            return ref
        
        # 相對引用
        if ref == "前項":
            # 需要找到前一項
            if current_item_no:
                prev_item = self._get_previous_item(current_clause_no, current_item_no)
                if prev_item:
                    return f"{current_clause_no}第{prev_item}項"
            return f"{current_clause_no}前項"
        
        elif ref == "前款":
            # 需要找到前一款
            return f"{current_clause_no}前款"
        
        elif ref == "前條":
            prev_clause = self._get_previous_clause(current_clause_no)
            if prev_clause:
                return prev_clause
            return ref
        
        elif ref in ["本條", "本項", "本款"]:
            return f"{current_clause_no}{ref[1:]}"
        
        return ref
    
    def _get_previous_clause(self, current_clause_no: str) -> Optional[str]:
        """獲取前一條文"""
        # 提取數字
        match = re.search(r"第([一二三四五六七八九十百]+)條", current_clause_no)
        if not match:
            return None
        
        # TODO: 實現中文數字轉換
        return None
    
    def _get_previous_item(self, clause_no: str, current_item_no: str) -> Optional[str]:
        """獲取前一項"""
        # 簡單映射
        item_map = {
            "二": "一", "三": "二", "四": "三", "五": "四",
            "六": "五", "七": "六", "八": "七", "九": "八", "十": "九"
        }
        return item_map.get(current_item_no)


# ==================== 數據結構 ====================

@dataclass
class SubItem:
    """款項結構"""
    subitem_no: str
    context: str
    raw_text: str
    reference_clauses: List[str]
    
    # 新增字段
    action_keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Item:
    """項目結構"""
    item_no: str
    context: str
    raw_text: str
    sub_items: List[SubItem]
    reference_clauses: List[str]
    intent_ids: List[str]
    
    # 新增字段
    action_keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "item_no": self.item_no,
            "context": self.context,
            "raw_text": self.raw_text,
            "sub_items": [si.to_dict() for si in self.sub_items],
            "reference_clauses": self.reference_clauses,
            "intent_ids": self.intent_ids,
            "action_keywords": self.action_keywords,
            "entities": self.entities
        }


@dataclass
class Clause:
    """條文主體結構"""
    clause_no: str
    clause_title: str
    clause_id: str
    context: str
    raw_text: str
    items: List[Item]
    reference_clauses: List[str]
    intent_ids: List[str]
    
    # 新增字段
    insurance_types: List[str] = field(default_factory=list)
    clause_function: str = ""
    action_keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "clause_no": self.clause_no,
            "clause_title": self.clause_title,
            "clause_id": self.clause_id,
            "context": self.context,
            "raw_text": self.raw_text,
            "items": [item.to_dict() for item in self.items],
            "reference_clauses": self.reference_clauses,
            "intent_ids": self.intent_ids,
            "insurance_types": self.insurance_types,
            "clause_function": self.clause_function,
            "action_keywords": self.action_keywords,
            "entities": self.entities
        }


@dataclass
class Chunk:
    """完整的 chunk 結構"""
    chunk_id: str
    chapter_no: Optional[str]
    chapter_title: Optional[str]
    clause: Clause
    
    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "chapter_no": self.chapter_no,
            "chapter_title": self.chapter_title,
            "clause": self.clause.to_dict(),
            # 向後兼容
            "context": self.clause.context,
            "raw_text": self.clause.raw_text,
            "items": [item.to_dict() for item in self.clause.items],
            "intent_ids": self.clause.intent_ids,
            "reference_clauses": self.clause.reference_clauses
        }


# ==================== 輔助函數 ====================

def clean_text(text: str, remove_numbers: bool = True) -> str:
    """清理文本"""
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_numbers:
        text = re.sub(r'^[一二三四五六七八九十百]+、\s*', '', text)
        text = re.sub(r'^(?:\(|（)[一二三四五六七八九十百0-9a-zA-Z]+(?:\)|）)\s*', '', text)
    
    return text


def detect_reference_clauses(text: str) -> List[str]:
    """檢測文本中的引用關係"""
    matches = REFERENCE_PATTERN.findall(text)
    seen = set()
    result = []
    for ref in matches:
        if ref not in seen:
            seen.add(ref)
            result.append(ref)
    return result


def parse_subitems(text: str, clause_no: str, item_no: str) -> List[SubItem]:
    """解析款項"""
    matches = list(SUBITEM_PATTERN.finditer(text))
    if not matches:
        return []
    
    subitems = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        subitem_text = text[start:end].strip()
        
        # 語義分析
        action_keywords = SemanticClassifier.extract_action_keywords(subitem_text)
        entities = SemanticClassifier.extract_entities(subitem_text)
        
        subitems.append(SubItem(
            subitem_no=match.group(1),
            context=subitem_text,
            raw_text=clean_text(subitem_text, remove_numbers=True),
            reference_clauses=detect_reference_clauses(subitem_text),
            action_keywords=action_keywords,
            entities=entities
        ))
    
    return subitems


def parse_items(text: str, clause_no: str, clause_title: str) -> List[Item]:
    """解析項目"""
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return []
    
    items = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        item_text = text[start:end].strip()
        item_no = match.group(1)
        
        # 解析子款項
        sub_items = parse_subitems(item_text, clause_no, item_no)
        
        # 語義分析
        action_keywords = SemanticClassifier.extract_action_keywords(item_text)
        entities = SemanticClassifier.extract_entities(item_text)
        
        items.append(Item(
            item_no=item_no,
            context=item_text,
            raw_text=clean_text(item_text, remove_numbers=True),
            sub_items=sub_items,
            reference_clauses=detect_reference_clauses(item_text),
            intent_ids=[],
            action_keywords=action_keywords,
            entities=entities
        ))
    
    return items


def extract_chapter_info(text: str, clause_start: int) -> Tuple[Optional[str], Optional[str]]:
    """提取章節信息"""
    chapters = list(CHAPTER_PATTERN.finditer(text))
    
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
    """從 PDF 生成結構化 chunks（V2 增強版）"""
    text = load_pdf().strip()
    
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
        items = parse_items(clause_body, clause_no, clause_title)
        
        # 語義分析
        insurance_types = SemanticClassifier.classify_insurance_type(clause_body, clause_title)
        clause_function = SemanticClassifier.classify_clause_function(clause_body, clause_title)
        action_keywords = SemanticClassifier.extract_action_keywords(clause_body)
        entities = SemanticClassifier.extract_entities(clause_body)
        
        # 構建條文對象
        clause = Clause(
            clause_no=clause_no,
            clause_title=clause_title,
            clause_id=clause_id,
            context=clause_body,
            raw_text=clean_text(clause_body, remove_numbers=True),
            items=items,
            reference_clauses=detect_reference_clauses(clause_body),
            intent_ids=[],
            insurance_types=insurance_types,
            clause_function=clause_function,
            action_keywords=action_keywords,
            entities=entities
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
    """保存 chunks 到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": {
            "total_chunks": len(chunks),
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "version": "2.0",
            "enhancements": [
                "語義分類（保險類型、條文功能）",
                "動作關鍵詞提取",
                "實體提取（時間、金額、次數）",
                "增強的引用關係"
            ]
        },
        "chunks": [chunk.to_dict() for chunk in chunks]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已生成 {len(chunks)} 個 chunks（V2），保存至 {output_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("🔄 開始生成 chunks（V2 增強版）...")
    
    chunks = generate_chunks_from_pdf()
    
    output_path = os.path.join(INDEX_DIR, "chunks_structured_v2.json")
    save_chunks(chunks, output_path)
    
    # 打印統計
    total_items = sum(len(chunk.clause.items) for chunk in chunks)
    insurance_types_count = {}
    clause_functions_count = {}
    
    for chunk in chunks:
        for itype in chunk.clause.insurance_types:
            insurance_types_count[itype] = insurance_types_count.get(itype, 0) + 1
        
        func = chunk.clause.clause_function
        clause_functions_count[func] = clause_functions_count.get(func, 0) + 1
    
    print(f"\n📊 統計:")
    print(f"   - 總條文數: {len(chunks)}")
    print(f"   - 總項目數: {total_items}")
    
    print(f"\n🏷️  保險類型分布:")
    for itype, count in sorted(insurance_types_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {itype}: {count}")
    
    print(f"\n📋 條文功能分布:")
    for func, count in sorted(clause_functions_count.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {func}: {count}")
