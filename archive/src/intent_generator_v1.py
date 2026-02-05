"""
Intent Generator - 從 chunks 生成意圖並建立索引

主要功能:
1. 使用 LLM 從條文生成結構化意圖
2. 提取使用者可能的查詢場景
3. 識別條件、例外、引用關係
4. 建立意圖嵌入索引以供檢索
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv
from config import INDEX_DIR
import time

# ==================== 初始化 ====================

load_dotenv()
client = OpenAI()


# ==================== 數據結構 ====================

@dataclass
class Intent:
    """意圖結構"""
    intent_id: str  # intent_0001
    clause_id: str  # 第一條_契約之構成
    item_no: Optional[str]  # 一、二、三（如果是針對特定項目）
    subitem_no: Optional[str]  # (一)、(二)（如果是針對特定款項）
    
    # 核心內容
    user_query: str  # 使用者可能的問題
    excerpt: str  # 相關條文摘錄
    
    # 詳細信息
    conditions: List[str]  # 適用條件
    exceptions: List[str]  # 例外情況
    referenced_clauses: List[str]  # 引用的其他條文
    
    # 分類
    category: str  # 如：賠償範圍、理賠條件、除外責任
    
    def to_dict(self):
        return asdict(self)


# ==================== LLM 提示詞 ====================

INTENT_GENERATION_PROMPT = """你是保險條款分析專家。請分析以下條文，生成 3-5 個使用者可能會問的問題（意圖），並提取相關信息。

條文信息：
章節：{chapter_info}
條文編號：{clause_no}
條文標題：{clause_title}
條文內容：
{context}

請以 JSON 格式返回，包含以下字段：
{{
  "intents": [
    {{
      "user_query": "使用者可能的問題",
      "excerpt": "回答該問題的關鍵條文摘錄",
      "conditions": ["適用條件1", "適用條件2"],
      "exceptions": ["例外情況1", "例外情況2"],
      "referenced_clauses": ["引用的其他條文"],
      "category": "分類（如：賠償範圍、理賠條件、除外責任、申請流程等）"
    }}
  ]
}}

注意事項：
1. user_query 應該是自然語言問題，例如："什麼情況下可以申請旅遊延誤賠償？"
2. excerpt 應該精確摘錄回答問題的關鍵部分（不超過100字）
3. conditions 是觸發該條款的條件
4. exceptions 是該條款不適用的情況
5. referenced_clauses 應該是完整的條文引用，例如："第二十七條第一項第二款"
6. 如果某個字段不適用，請使用空列表 []
7. 只返回 JSON，不要包含其他說明文字

直接返回 JSON："""


ITEM_INTENT_GENERATION_PROMPT = """你是保險條款分析專家。請分析以下項目內容，生成 1-2 個使用者可能會問的問題。

母條文：{clause_no} {clause_title}
項目編號：{item_no}
項目內容：
{item_context}

請以 JSON 格式返回：
{{
  "intents": [
    {{
      "user_query": "使用者可能的問題",
      "excerpt": "回答該問題的關鍵內容摘錄",
      "conditions": ["適用條件"],
      "exceptions": ["例外情況"],
      "referenced_clauses": ["引用的其他條文"],
      "category": "分類"
    }}
  ]
}}

只返回 JSON："""


# ==================== LLM 調用 ====================

def call_llm_for_intents(prompt: str, max_retries: int = 3) -> List[Dict]:
    """
    調用 LLM 生成意圖
    
    Args:
        prompt: 提示詞
        max_retries: 最大重試次數
    
    Returns:
        意圖列表
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "你是專業的保險條款分析專家，擅長從條文中提取使用者意圖。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # 移除可能的 markdown 代碼塊標記
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # 解析 JSON
            result = json.loads(content)
            return result.get("intents", [])
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 解析錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"❌ 無法解析 LLM 響應: {content[:200]}")
                return []
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️  LLM 調用錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(2)
    
    return []


# ==================== 意圖生成 ====================

def generate_clause_intents(chunk: Dict, intent_id_counter: List[int]) -> List[Intent]:
    """
    為條文生成意圖
    
    Args:
        chunk: 條文 chunk
        intent_id_counter: 意圖 ID 計數器（列表包裝以支持引用傳遞）
    
    Returns:
        Intent 列表
    """
    clause = chunk["clause"]
    chapter_info = f"{chunk.get('chapter_no', '')} {chunk.get('chapter_title', '')}" if chunk.get('chapter_no') else "無章節"
    
    # 構建提示詞
    prompt = INTENT_GENERATION_PROMPT.format(
        chapter_info=chapter_info,
        clause_no=clause["clause_no"],
        clause_title=clause["clause_title"],
        context=clause["context"]
    )
    
    # 調用 LLM
    llm_intents = call_llm_for_intents(prompt)
    
    # 轉換為 Intent 對象
    intents = []
    for llm_intent in llm_intents:
        intent_id = f"intent_{intent_id_counter[0]:04d}"
        intent_id_counter[0] += 1
        
        intents.append(Intent(
            intent_id=intent_id,
            clause_id=clause["clause_id"],
            item_no=None,
            subitem_no=None,
            user_query=llm_intent.get("user_query", ""),
            excerpt=llm_intent.get("excerpt", ""),
            conditions=llm_intent.get("conditions", []),
            exceptions=llm_intent.get("exceptions", []),
            referenced_clauses=llm_intent.get("referenced_clauses", []),
            category=llm_intent.get("category", "其他")
        ))
    
    return intents


def generate_item_intents(chunk: Dict, item: Dict, intent_id_counter: List[int]) -> List[Intent]:
    """
    為項目生成意圖
    
    Args:
        chunk: 條文 chunk
        item: 項目數據
        intent_id_counter: 意圖 ID 計數器
    
    Returns:
        Intent 列表
    """
    clause = chunk["clause"]
    
    # 構建提示詞
    prompt = ITEM_INTENT_GENERATION_PROMPT.format(
        clause_no=clause["clause_no"],
        clause_title=clause["clause_title"],
        item_no=item["item_no"],
        item_context=item["context"]
    )
    
    # 調用 LLM
    llm_intents = call_llm_for_intents(prompt)
    
    # 轉換為 Intent 對象
    intents = []
    for llm_intent in llm_intents:
        intent_id = f"intent_{intent_id_counter[0]:04d}"
        intent_id_counter[0] += 1
        
        intents.append(Intent(
            intent_id=intent_id,
            clause_id=clause["clause_id"],
            item_no=item["item_no"],
            subitem_no=None,
            user_query=llm_intent.get("user_query", ""),
            excerpt=llm_intent.get("excerpt", ""),
            conditions=llm_intent.get("conditions", []),
            exceptions=llm_intent.get("exceptions", []),
            referenced_clauses=llm_intent.get("referenced_clauses", []),
            category=llm_intent.get("category", "其他")
        ))
    
    return intents


def generate_all_intents(chunks: List[Dict], 
                         generate_for_items: bool = True) -> List[Intent]:
    """
    為所有 chunks 生成意圖
    
    Args:
        chunks: Chunk 列表
        generate_for_items: 是否也為子項目生成意圖
    
    Returns:
        所有 Intent 列表
    """
    all_intents = []
    intent_id_counter = [1]  # 使用列表以支持引用傳遞
    
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, 1):
        clause = chunk["clause"]
        print(f"🔄 處理 [{i}/{total_chunks}]: {clause['clause_no']} {clause['clause_title']}")
        
        # 生成條文級別的意圖
        clause_intents = generate_clause_intents(chunk, intent_id_counter)
        all_intents.extend(clause_intents)
        
        # 如果需要，為每個項目生成意圖
        if generate_for_items and clause.get("items"):
            for item in clause["items"]:
                item_intents = generate_item_intents(chunk, item, intent_id_counter)
                all_intents.extend(item_intents)
                
                # 將 intent_id 添加到 item 中
                item["intent_ids"] = [intent.intent_id for intent in item_intents]
        
        # 將條文級別的 intent_id 添加到 clause 中
        clause["intent_ids"] = [intent.intent_id for intent in clause_intents]
        
        # 控制請求頻率
        time.sleep(0.5)
    
    print(f"\n✅ 總共生成 {len(all_intents)} 個意圖")
    return all_intents


# ==================== 保存函數 ====================

def save_intents(intents: List[Intent], output_path: str):
    """保存意圖到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": {
            "total_intents": len(intents),
            "generated_at": __import__('datetime').datetime.now().isoformat()
        },
        "intents": [intent.to_dict() for intent in intents]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存 {len(intents)} 個意圖至 {output_path}")


def save_chunks_with_intents(chunks: List[Dict], output_path: str):
    """保存包含意圖 ID 的 chunks"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": {
            "total_chunks": len(chunks),
            "generated_at": __import__('datetime').datetime.now().isoformat()
        },
        "chunks": chunks
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存 chunks（含意圖）至 {output_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("🔄 開始生成意圖...")
    
    # 載入 chunks
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured.json")
    
    if not os.path.exists(chunks_path):
        print(f"❌ 找不到 chunks 文件: {chunks_path}")
        print("請先運行 chunk_generator.py 生成 chunks")
        exit(1)
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data["chunks"]
    print(f"📥 已載入 {len(chunks)} 個 chunks")
    
    # 生成意圖
    intents = generate_all_intents(chunks, generate_for_items=True)
    
    # 保存意圖
    intents_path = os.path.join(INDEX_DIR, "intents.json")
    save_intents(intents, intents_path)
    
    # 保存更新後的 chunks
    chunks_with_intents_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
    save_chunks_with_intents(chunks, chunks_with_intents_path)
    
    # 統計信息
    categories = {}
    for intent in intents:
        cat = intent.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 意圖分類統計:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {cat}: {count}")
