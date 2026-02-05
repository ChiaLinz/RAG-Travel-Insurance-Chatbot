"""
Intent Generator V2 - 優化版意圖生成器

主要優化：
1. 針對性問法生成（區分「延誤」vs「遺失」vs「損失」）
2. 對比式意圖（幫助用戶區分相似概念）
3. 多樣化問法（同義詞變體、口語化表達）
4. 負向意圖（明確什麼不能做）
5. 條件組合意圖（複雜場景）
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
    intent_id: str
    clause_id: str
    item_no: Optional[str]
    subitem_no: Optional[str]
    
    # 核心內容
    user_query: str
    excerpt: str
    
    # 詳細信息
    conditions: List[str]
    exceptions: List[str]
    referenced_clauses: List[str]
    
    # 分類
    category: str
    
    # V2 新增字段
    query_type: str = "直接查詢"  # 直接查詢、對比查詢、條件查詢、負向查詢
    semantic_tags: List[str] = None  # 語義標籤（延誤、遺失、損失等）
    difficulty: str = "簡單"  # 簡單、中等、複雜
    
    def __post_init__(self):
        if self.semantic_tags is None:
            self.semantic_tags = []
    
    def to_dict(self):
        return asdict(self)


# ==================== LLM 提示詞（V2 增強版）====================

INTENT_GENERATION_PROMPT_V2 = """你是保險條款分析專家。請仔細分析以下條文，生成 5-8 個不同類型的用戶可能問題（意圖）。

條文信息：
章節：{chapter_info}
條文編號：{clause_no}
條文標題：{clause_title}
條文功能：{clause_function}
保險類型：{insurance_types}
動作關鍵詞：{action_keywords}
條文內容：
{context}

**重要提示**：
1. 必須生成多樣化的問法，包括：
   - 直接查詢（最常見的問法）
   - 口語化表達（用戶實際會怎麼問）
   - 特定場景（具體情境下的問題）
   - 負向查詢（什麼情況下不能/不會）

2. 特別注意區分相似概念：
   - "延誤" vs "遺失" vs "損失" vs "取消" vs "更改"
   - 每個詞的問法都要明確區分

3. 如果是「不保事項」條文，必須生成負向意圖：
   - "哪些情況不理賠？"
   - "什麼時候不能申請？"

請以 JSON 格式返回：
{{
  "intents": [
    {{
      "user_query": "用戶可能的問題（要自然、口語化）",
      "excerpt": "回答該問題的關鍵條文摘錄（不超過100字）",
      "conditions": ["適用條件1", "適用條件2"],
      "exceptions": ["例外情況1"],
      "referenced_clauses": ["引用的其他條文"],
      "category": "分類（賠償範圍/理賠條件/除外責任/申請流程/定義說明）",
      "query_type": "查詢類型（直接查詢/對比查詢/條件查詢/負向查詢）",
      "semantic_tags": ["語義標籤，如：延誤、遺失、竊盜等"],
      "difficulty": "難度（簡單/中等/複雜）"
    }}
  ]
}}

**範例**（假設是「行李損失保險承保範圍」條文）：
{{
  "intents": [
    {{
      "user_query": "行李被偷了可以理賠嗎？",
      "excerpt": "因竊盜、強盜與搶奪導致行李遺失可以理賠",
      "conditions": ["竊盜、強盜或搶奪", "置於行李箱內", "海外旅行期間"],
      "exceptions": [],
      "referenced_clauses": [],
      "category": "賠償範圍",
      "query_type": "直接查詢",
      "semantic_tags": ["遺失", "竊盜"],
      "difficulty": "簡單"
    }},
    {{
      "user_query": "行李被航空公司弄丟了怎麼辦？",
      "excerpt": "託運行李因業者處理失當導致遺失可理賠",
      "conditions": ["託運行李", "領有託運單", "業者處理失當"],
      "exceptions": [],
      "referenced_clauses": [],
      "category": "賠償範圍",
      "query_type": "直接查詢",
      "semantic_tags": ["遺失", "託運"],
      "difficulty": "簡單"
    }},
    {{
      "user_query": "行李延誤和行李遺失有什麼不同？",
      "excerpt": "行李遺失是指毀損、滅失；延誤是指未能及時領取",
      "conditions": [],
      "exceptions": [],
      "referenced_clauses": ["第三十六條"],
      "category": "定義說明",
      "query_type": "對比查詢",
      "semantic_tags": ["遺失", "延誤", "對比"],
      "difficulty": "中等"
    }},
    {{
      "user_query": "哪些東西丟了不能理賠？",
      "excerpt": "商業用品、貨幣、證券等不在理賠範圍",
      "conditions": [],
      "exceptions": ["商業用品", "貨幣", "證券"],
      "referenced_clauses": ["第四十條"],
      "category": "除外責任",
      "query_type": "負向查詢",
      "semantic_tags": ["遺失", "不保"],
      "difficulty": "簡單"
    }}
  ]
}}

只返回 JSON，不要其他說明："""


ITEM_INTENT_GENERATION_PROMPT_V2 = """你是保險條款分析專家。請分析以下項目內容，生成 2-3 個用戶可能問題。

母條文：{clause_no} {clause_title}
條文功能：{clause_function}
項目編號：{item_no}
項目內容：
{item_context}

動作關鍵詞：{action_keywords}

**注意事項**：
1. 問法要針對這個特定項目
2. 區分不同的動作詞（延誤/遺失/損失/取消）
3. 如果是不保事項的項目，要生成負向問法

請以 JSON 格式返回：
{{
  "intents": [
    {{
      "user_query": "針對這個項目的具體問題",
      "excerpt": "關鍵內容摘錄",
      "conditions": ["條件"],
      "exceptions": ["例外"],
      "referenced_clauses": ["引用"],
      "category": "分類",
      "query_type": "查詢類型",
      "semantic_tags": ["語義標籤"],
      "difficulty": "難度"
    }}
  ]
}}

只返回 JSON："""


# ==================== 對比意圖生成 ====================

COMPARISON_INTENT_TEMPLATE = {
    "行李延誤_vs_行李損失": {
        "user_query": "行李延誤和行李遺失有什麼差別？什麼時候算延誤，什麼時候算遺失？",
        "excerpt": "行李延誤是指抵達6小時後仍未領得；行李損失是指毀損、滅失或遺失",
        "category": "定義說明",
        "query_type": "對比查詢",
        "semantic_tags": ["延誤", "遺失", "對比"],
        "difficulty": "中等",
        "related_clauses": ["第三十六條", "第三十九條"]
    },
    "班機延誤_vs_旅程取消": {
        "user_query": "班機延誤和旅程取消有什麼不同？分別在什麼情況下理賠？",
        "excerpt": "班機延誤是班機晚點；旅程取消是在出發前因特定事由取消整個行程",
        "category": "定義說明",
        "query_type": "對比查詢",
        "semantic_tags": ["延誤", "取消", "對比"],
        "difficulty": "中等",
        "related_clauses": ["第二十七條", "第三十條"]
    },
    "旅程取消_vs_旅程更改": {
        "user_query": "旅程取消和旅程更改有何差別？",
        "excerpt": "旅程取消是出發前全部取消；旅程更改是旅行中因故變更行程",
        "category": "定義說明",
        "query_type": "對比查詢",
        "semantic_tags": ["取消", "更改", "對比"],
        "difficulty": "中等",
        "related_clauses": ["第二十七條", "第三十三條"]
    },
    "竊盜_vs_處理失當": {
        "user_query": "行李被偷和被航空公司弄丟，理賠有什麼不同？",
        "excerpt": "竊盜需報警並取得報案證明；處理失當需業者出具事故證明",
        "category": "申請流程",
        "query_type": "對比查詢",
        "semantic_tags": ["竊盜", "遺失", "對比"],
        "difficulty": "中等",
        "related_clauses": ["第三十九條", "第四十二條", "第四十三條"]
    }
}


def generate_comparison_intents(chunks: List[Dict], intent_id_counter: List[int]) -> List[Intent]:
    """生成對比意圖"""
    intents = []
    
    for comp_key, comp_data in COMPARISON_INTENT_TEMPLATE.items():
        intent_id = f"intent_{intent_id_counter[0]:04d}"
        intent_id_counter[0] += 1
        
        # 找到相關條文
        related_clauses = comp_data.get("related_clauses", [])
        main_clause_id = None
        
        for chunk in chunks:
            clause_no = chunk["clause"]["clause_no"]
            if related_clauses and clause_no in related_clauses[0]:
                main_clause_id = chunk["clause"]["clause_id"]
                break
        
        if not main_clause_id:
            main_clause_id = chunks[0]["clause"]["clause_id"]  # 默認第一條
        
        intents.append(Intent(
            intent_id=intent_id,
            clause_id=main_clause_id,
            item_no=None,
            subitem_no=None,
            user_query=comp_data["user_query"],
            excerpt=comp_data["excerpt"],
            conditions=[],
            exceptions=[],
            referenced_clauses=related_clauses,
            category=comp_data["category"],
            query_type=comp_data["query_type"],
            semantic_tags=comp_data["semantic_tags"],
            difficulty=comp_data["difficulty"]
        ))
    
    print(f"✅ 已生成 {len(intents)} 個對比意圖")
    return intents


# ==================== LLM 調用 ====================

def call_llm_for_intents(prompt: str, max_retries: int = 3) -> List[Dict]:
    """調用 LLM 生成意圖"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "你是專業的保險條款分析專家，擅長從條文中提取多樣化的用戶意圖。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,  # 稍微提高創造性
                max_tokens=2500
            )
            
            content = response.choices[0].message.content.strip()
            
            # 移除 markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            return result.get("intents", [])
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 解析錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"❌ 無法解析: {content[:200]}")
                return []
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️  LLM 錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(2)
    
    return []


# ==================== 意圖生成 ====================

def generate_clause_intents(chunk: Dict, intent_id_counter: List[int]) -> List[Intent]:
    """為條文生成意圖（V2 增強版）"""
    clause = chunk["clause"]
    chapter_info = f"{chunk.get('chapter_no', '')} {chunk.get('chapter_title', '')}" if chunk.get('chapter_no') else "無章節"
    
    # 構建增強的提示詞
    prompt = INTENT_GENERATION_PROMPT_V2.format(
        chapter_info=chapter_info,
        clause_no=clause["clause_no"],
        clause_title=clause["clause_title"],
        clause_function=clause.get("clause_function", "一般規定"),
        insurance_types=", ".join(clause.get("insurance_types", ["其他"])),
        action_keywords=", ".join(clause.get("action_keywords", [])),
        context=clause["context"]
    )
    
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
            category=llm_intent.get("category", "其他"),
            query_type=llm_intent.get("query_type", "直接查詢"),
            semantic_tags=llm_intent.get("semantic_tags", []),
            difficulty=llm_intent.get("difficulty", "簡單")
        ))
    
    return intents


def generate_item_intents(chunk: Dict, item: Dict, intent_id_counter: List[int]) -> List[Intent]:
    """為項目生成意圖（V2 增強版）"""
    clause = chunk["clause"]
    
    prompt = ITEM_INTENT_GENERATION_PROMPT_V2.format(
        clause_no=clause["clause_no"],
        clause_title=clause["clause_title"],
        clause_function=clause.get("clause_function", "一般規定"),
        item_no=item["item_no"],
        item_context=item["context"],
        action_keywords=", ".join(item.get("action_keywords", []))
    )
    
    llm_intents = call_llm_for_intents(prompt)
    
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
            category=llm_intent.get("category", "其他"),
            query_type=llm_intent.get("query_type", "直接查詢"),
            semantic_tags=llm_intent.get("semantic_tags", []),
            difficulty=llm_intent.get("difficulty", "簡單")
        ))
    
    return intents


def generate_all_intents(chunks: List[Dict], 
                         generate_for_items: bool = True,
                         generate_comparisons: bool = True) -> List[Intent]:
    """生成所有意圖（V2 增強版）"""
    all_intents = []
    intent_id_counter = [1]
    
    total_chunks = len(chunks)
    
    # 1. 生成條文和項目意圖
    for i, chunk in enumerate(chunks, 1):
        clause = chunk["clause"]
        print(f"🔄 處理 [{i}/{total_chunks}]: {clause['clause_no']} {clause['clause_title']}")
        
        # 條文級別意圖
        clause_intents = generate_clause_intents(chunk, intent_id_counter)
        all_intents.extend(clause_intents)
        
        # 項目級別意圖
        if generate_for_items and clause.get("items"):
            for item in clause["items"]:
                item_intents = generate_item_intents(chunk, item, intent_id_counter)
                all_intents.extend(item_intents)
                item["intent_ids"] = [intent.intent_id for intent in item_intents]
        
        clause["intent_ids"] = [intent.intent_id for intent in clause_intents]
        
        time.sleep(0.5)
    
    # 2. 生成對比意圖
    if generate_comparisons:
        print("\n🔄 生成對比意圖...")
        comparison_intents = generate_comparison_intents(chunks, intent_id_counter)
        all_intents.extend(comparison_intents)
    
    print(f"\n✅ 總共生成 {len(all_intents)} 個意圖")
    
    # 統計
    query_types = {}
    for intent in all_intents:
        qt = intent.query_type
        query_types[qt] = query_types.get(qt, 0) + 1
    
    print(f"\n📊 意圖類型分布:")
    for qt, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {qt}: {count}")
    
    return all_intents


# ==================== 保存函數 ====================

def save_intents(intents: List[Intent], output_path: str):
    """保存意圖到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": {
            "total_intents": len(intents),
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "version": "2.0",
            "enhancements": [
                "針對性問法（區分延誤/遺失/損失）",
                "對比式意圖",
                "多樣化問法",
                "負向意圖",
                "語義標籤"
            ]
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
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "version": "2.0"
        },
        "chunks": chunks
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存 chunks（V2，含意圖）至 {output_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("🔄 開始生成意圖（V2 增強版）...")
    
    # 載入 V2 chunks
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured_v2.json")
    
    if not os.path.exists(chunks_path):
        print(f"❌ 找不到 V2 chunks 文件: {chunks_path}")
        print("請先運行 chunk_generator_v2.py")
        exit(1)
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data["chunks"]
    print(f"📥 已載入 {len(chunks)} 個 chunks（V2）")
    
    # 生成意圖
    intents = generate_all_intents(
        chunks, 
        generate_for_items=True,
        generate_comparisons=True  # 啟用對比意圖
    )
    
    # 保存
    intents_path = os.path.join(INDEX_DIR, "intents_v2.json")
    save_intents(intents, intents_path)
    
    chunks_with_intents_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents_v2.json")
    save_chunks_with_intents(chunks, chunks_with_intents_path)
    
    # 詳細統計
    categories = {}
    semantic_tags_count = {}
    
    for intent in intents:
        cat = intent.category
        categories[cat] = categories.get(cat, 0) + 1
        
        for tag in intent.semantic_tags:
            semantic_tags_count[tag] = semantic_tags_count.get(tag, 0) + 1
    
    print("\n📊 意圖分類統計:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {cat}: {count}")
    
    print("\n🏷️  語義標籤統計:")
    for tag, count in sorted(semantic_tags_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {tag}: {count}")
