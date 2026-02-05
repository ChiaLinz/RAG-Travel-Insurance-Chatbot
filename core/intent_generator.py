"""
Intent Generator - 優化提示詞版本

主要改進：
1. Few-shot Learning（添加正反例）
2. 更嚴格的語義標籤要求
3. 後處理驗證與自動修正
4. 質量評分機制
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv
from config import INDEX_DIR
import time
import re


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
    
    # 字段
    query_type: str = "直接查詢"
    semantic_tags: List[str] = None
    difficulty: str = "簡單"
    quality_score: float = 1.0  # 新增：質量評分
    
    def __post_init__(self):
        if self.semantic_tags is None:
            self.semantic_tags = []
    
    def to_dict(self):
        return asdict(self)


# ==================== Few-shot 範例 ====================

FEW_SHOT_EXAMPLES = """
【範例 1 - 正確】

條文：第三十九條 行李損失保險承保範圍
動作關鍵詞：["遺失", "竊盜", "損失"]
內容：被保險人於海外旅行期間內，因下列事故致其所擁有且置於行李箱、手提箱或類似容器內之個人物品遭受毀損或滅失...
一、竊盜、強盜與搶奪。
二、交由所搭乘之公共交通工具業者託運且領有託運行李領取單之隨行託運行李，因該公共交通工具業者處理失當所致之毀損、滅失或遺失。

✅ 正確的意圖生成：
{
  "user_query": "行李被偷了可以理賠嗎？",
  "excerpt": "因竊盜、強盜與搶奪導致行李遺失，本公司依保險契約給付保險金",
  "conditions": ["竊盜、強盜或搶奪", "海外旅行期間", "置於行李箱內"],
  "exceptions": [],
  "referenced_clauses": ["第四十條", "第四十三條"],
  "category": "賠償範圍",
  "query_type": "直接查詢",
  "semantic_tags": ["遺失", "竊盜"],  ← 關鍵！必須包含動作關鍵詞
  "difficulty": "簡單"
}

{
  "user_query": "行李被航空公司弄丟了怎麼辦？",
  "excerpt": "託運行李因業者處理失當導致遺失可理賠",
  "conditions": ["託運行李", "領有託運單", "業者處理失當"],
  "exceptions": [],
  "referenced_clauses": ["第四十三條"],
  "category": "賠償範圍",
  "query_type": "直接查詢",
  "semantic_tags": ["遺失", "託運"],  ← 必須明確標註
  "difficulty": "簡單"
}

❌ 錯誤的意圖生成：
{
  "user_query": "行李問題怎麼理賠？",  ← 太寬泛
  "semantic_tags": ["理賠"],  ← 缺少「遺失」標籤
  ...
}

---

【範例 2 - 正確】

條文：第三十六條 行李延誤保險承保範圍
動作關鍵詞：["延誤"]
內容：被保險人於海外旅行期間內，其隨行託運並取得託運行李領取單之個人行李因公共交通工具業者之處理失當，致其在抵達目的地六小時後仍未領得時...

✅ 正確的意圖生成：
{
  "user_query": "行李延誤多久可以理賠？",
  "excerpt": "抵達目的地六小時後仍未領得行李可理賠",
  "conditions": ["隨行託運", "有託運單", "六小時後仍未領得"],
  "exceptions": [],
  "referenced_clauses": ["第三十七條", "第三十八條"],
  "category": "賠償範圍",
  "query_type": "直接查詢",
  "semantic_tags": ["延誤"],  ← 必須是「延誤」而非「遺失」
  "difficulty": "簡單"
}

❌ 錯誤的意圖生成：
{
  "user_query": "行李延誤多久可以理賠？",
  "semantic_tags": ["遺失"],  ← 錯誤！應該是「延誤」
  ...
}

---

【範例 3 - 對比意圖】

✅ 正確的對比意圖：
{
  "user_query": "行李延誤和行李遺失有什麼差別？",
  "excerpt": "延誤是指未能及時領取；遺失是指毀損、滅失",
  "conditions": [],
  "exceptions": [],
  "referenced_clauses": ["第三十六條", "第三十九條"],
  "category": "定義說明",
  "query_type": "對比查詢",
  "semantic_tags": ["延誤", "遺失", "對比"],  ← 必須包含兩個概念
  "difficulty": "中等"
}

---

【範例 4 - 負向意圖】

條文：第四十條 行李損失保險特別不保事項（物品）
條文功能：不保事項

✅ 正確的負向意圖：
{
  "user_query": "哪些東西丟了不能理賠？",
  "excerpt": "商業用品、貨幣、證券、珠寶、手機等不在理賠範圍",
  "conditions": [],
  "exceptions": ["商業用品", "貨幣", "證券"],
  "referenced_clauses": [],
  "category": "除外責任",
  "query_type": "負向查詢",
  "semantic_tags": ["遺失", "不保"],  ← 必須包含「不保」
  "difficulty": "簡單"
}

❌ 錯誤的負向意圖：
{
  "user_query": "哪些東西丟了不能理賠？",
  "semantic_tags": ["遺失"],  ← 缺少「不保」標籤
  ...
}
"""


# ==================== 優化提示詞 ====================

INTENT_GENERATION_PROMPT_= """你是保險條款分析專家。請仔細分析以下條文，生成 5-8 個高質量的用戶問題（意圖）。

條文信息：
條文編號：{clause_no}
條文標題：{clause_title}
條文功能：{clause_function}
保險類型：{insurance_types}
動作關鍵詞：{action_keywords}
條文內容：
{context}

**重要規則（必須遵守）**：

1. **語義標籤 (semantic_tags) 規則**：
   - 必須包含條文的「動作關鍵詞」：{action_keywords}
   - 如果條文功能是「不保事項」，必須包含 "不保"
   - 如果是對比查詢，必須包含兩個對比的概念
   
   範例：
   ✅ 正確："行李被偷" → semantic_tags: ["遺失", "竊盜"]
   ❌ 錯誤："行李被偷" → semantic_tags: ["理賠"]

2. **問法多樣化**：
   - 直接查詢：最常見的問法（如："XX可以理賠嗎？"）
   - 口語化：用戶實際會怎麼問（如："行李弄丟了怎麼辦？"）
   - 負向查詢：什麼不能/不會（如："哪些情況不理賠？"）
   
3. **特殊要求**：
   - 如果條文功能 = "不保事項"，至少生成 2 個負向意圖
   - 避免太寬泛的問法（如："保險怎麼賠？"）
   - 每個 user_query 必須明確、具體

4. **嚴格區分相似概念**：
   - "延誤" ≠ "遺失" ≠ "損失" ≠ "取消" ≠ "更改"
   - 每個詞的語義標籤都不同，不能混淆

{few_shot_examples}

請以 JSON 格式返回：
{{
  "intents": [
    {{
      "user_query": "具體、明確的問題",
      "excerpt": "關鍵條文摘錄（<100字）",
      "conditions": ["條件1", "條件2"],
      "exceptions": ["例外1"],
      "referenced_clauses": ["第XX條"],
      "category": "賠償範圍/理賠條件/除外責任/申請流程/定義說明",
      "query_type": "直接查詢/對比查詢/負向查詢/條件查詢",
      "semantic_tags": ["必須包含動作關鍵詞"],
      "difficulty": "簡單/中等/複雜"
    }}
  ]
}}

**檢查清單**（生成前請自我檢查）：
□ semantic_tags 是否包含動作關鍵詞？
□ 如果是不保事項，是否包含「不保」標籤？
□ user_query 是否夠具體？
□ 是否避免了與其他概念混淆？

只返回 JSON，不要其他說明："""


ITEM_INTENT_GENERATION_PROMPT_= """你是保險條款分析專家。請分析以下項目內容，生成 2-3 個精確的用戶問題。

母條文：{clause_no} {clause_title}
條文功能：{clause_function}
項目編號：{item_no}
動作關鍵詞：{action_keywords}
項目內容：
{item_context}

**必須遵守的規則**：
1. semantic_tags 必須包含：{action_keywords}
2. 如果條文功能 = "不保事項"，必須包含 "不保"
3. 問法要針對這個特定項目

請以 JSON 格式返回：
{{
  "intents": [
    {{
      "user_query": "針對此項目的具體問題",
      "excerpt": "關鍵內容",
      "conditions": ["條件"],
      "exceptions": ["例外"],
      "referenced_clauses": ["引用"],
      "category": "分類",
      "query_type": "查詢類型",
      "semantic_tags": ["必須包含動作關鍵詞"],
      "difficulty": "難度"
    }}
  ]
}}

只返回 JSON："""


# ==================== 質量驗證器 ====================

class IntentQualityValidator:
    """Intent 質量驗證器"""
    
    @staticmethod
    def validate_and_fix(intent: Dict, clause: Dict) -> tuple[Dict, float]:
        """
        驗證並修正 intent
        
        Returns:
            (修正後的 intent, 質量評分)
        """
        quality_score = 1.0
        issues = []
        
        # 1. 檢查 semantic_tags 是否包含 action_keywords
        required_tags = clause.get("action_keywords", [])
        current_tags = intent.get("semantic_tags", [])
        
        missing_tags = [tag for tag in required_tags if tag not in current_tags]
        if missing_tags:
            intent["semantic_tags"] = current_tags + missing_tags
            quality_score -= 0.2
            issues.append(f"缺少動作標籤: {missing_tags}")
        
        # 2. 檢查不保事項
        if clause.get("clause_function") == "不保事項":
            if "不保" not in current_tags:
                intent["semantic_tags"].append("不保")
                quality_score -= 0.15
                issues.append("不保事項缺少'不保'標籤")
        
        # 3. 檢查 user_query 是否太寬泛
        vague_patterns = ["怎麼辦", "如何", "可以嗎"]
        query = intent.get("user_query", "")
        if len(query) < 8 or all(p not in query for p in vague_patterns):
            # 查詢可能太簡單或太寬泛
            if len(query) < 6:
                quality_score -= 0.1
                issues.append("查詢太短")
        
        # 4. 檢查對比查詢
        if intent.get("query_type") == "對比查詢":
            if len(current_tags) < 2:
                quality_score -= 0.2
                issues.append("對比查詢應包含至少2個語義標籤")
        
        # 5. 檢查是否混淆概念
        confusing_pairs = [
            (["延誤"], ["遺失", "損失"]),
            (["遺失"], ["延誤"]),
            (["取消"], ["更改"]),
        ]
        
        for group1, group2 in confusing_pairs:
            if any(t in current_tags for t in group1) and any(t in current_tags for t in group2):
                # 可能混淆（除非是對比查詢）
                if intent.get("query_type") != "對比查詢":
                    quality_score -= 0.3
                    issues.append(f"可能混淆概念: {group1} vs {group2}")
        
        # 記錄問題
        if issues:
            intent["validation_issues"] = issues
        
        return intent, quality_score


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
                        "content": "你是專業的保險條款分析專家。你必須嚴格遵守規則生成高質量的意圖，特別注意語義標籤的準確性。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # 降低溫度提高一致性
                max_tokens=3000
            )
            
            content = response.choices[0].message.content.strip()
            
            # 移除 markdown
            content = re.sub(r'^```json\s*\n?', '', content)
            content = re.sub(r'^```\s*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
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

def generate_clause_intents(chunk: Dict, 
                           intent_id_counter: List[int],
                           validator: IntentQualityValidator) -> List[Intent]:
    """為條文生成意圖（優化版）"""
    clause = chunk["clause"]
    
    # 構建 提示詞（包含 Few-shot）
    prompt = INTENT_GENERATION_PROMPT.format(
        clause_no=clause["clause_no"],
        clause_title=clause["clause_title"],
        clause_function=clause.get("clause_function", "一般規定"),
        insurance_types=", ".join(clause.get("insurance_types", ["其他"])),
        action_keywords=", ".join(clause.get("action_keywords", [])),
        context=clause["context"],
        few_shot_examples=FEW_SHOT_EXAMPLES
    )
    
    llm_intents = call_llm_for_intents(prompt)
    
    # 驗證並修正
    intents = []
    for llm_intent in llm_intents:
        # 驗證
        fixed_intent, quality_score = validator.validate_and_fix(llm_intent, clause)
        
        intent_id = f"intent_{intent_id_counter[0]:04d}"
        intent_id_counter[0] += 1
        
        intents.append(Intent(
            intent_id=intent_id,
            clause_id=clause["clause_id"],
            item_no=None,
            subitem_no=None,
            user_query=fixed_intent.get("user_query", ""),
            excerpt=fixed_intent.get("excerpt", ""),
            conditions=fixed_intent.get("conditions", []),
            exceptions=fixed_intent.get("exceptions", []),
            referenced_clauses=fixed_intent.get("referenced_clauses", []),
            category=fixed_intent.get("category", "其他"),
            query_type=fixed_intent.get("query_type", "直接查詢"),
            semantic_tags=fixed_intent.get("semantic_tags", []),
            difficulty=fixed_intent.get("difficulty", "簡單"),
            quality_score=quality_score
        ))
    
    return intents


def generate_item_intents(chunk: Dict, 
                         item: Dict, 
                         intent_id_counter: List[int],
                         validator: IntentQualityValidator) -> List[Intent]:
    """為項目生成意圖（優化版）"""
    clause = chunk["clause"]
    
    prompt = ITEM_INTENT_GENERATION_PROMPT.format(
        clause_no=clause["clause_no"],
        clause_title=clause["clause_title"],
        clause_function=clause.get("clause_function", "一般規定"),
        item_no=item["item_no"],
        action_keywords=", ".join(item.get("action_keywords", [])),
        item_context=item["context"]
    )
    
    llm_intents = call_llm_for_intents(prompt)
    
    intents = []
    for llm_intent in llm_intents:
        fixed_intent, quality_score = validator.validate_and_fix(llm_intent, clause)
        
        intent_id = f"intent_{intent_id_counter[0]:04d}"
        intent_id_counter[0] += 1
        
        intents.append(Intent(
            intent_id=intent_id,
            clause_id=clause["clause_id"],
            item_no=item["item_no"],
            subitem_no=None,
            user_query=fixed_intent.get("user_query", ""),
            excerpt=fixed_intent.get("excerpt", ""),
            conditions=fixed_intent.get("conditions", []),
            exceptions=fixed_intent.get("exceptions", []),
            referenced_clauses=fixed_intent.get("referenced_clauses", []),
            category=fixed_intent.get("category", "其他"),
            query_type=fixed_intent.get("query_type", "直接查詢"),
            semantic_tags=fixed_intent.get("semantic_tags", []),
            difficulty=fixed_intent.get("difficulty", "簡單"),
            quality_score=quality_score
        ))
    
    return intents


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
        
        related_clauses = comp_data.get("related_clauses", [])
        main_clause_id = chunks[0]["clause"]["clause_id"] if chunks else "第一條"
        
        for chunk in chunks:
            clause_no = chunk["clause"]["clause_no"]
            if related_clauses and clause_no in related_clauses[0]:
                main_clause_id = chunk["clause"]["clause_id"]
                break
        
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
            difficulty=comp_data["difficulty"],
            quality_score=1.0
        ))
    
    print(f"✅ 已生成 {len(intents)} 個對比意圖")
    return intents


# ==================== 主函數 ====================

def generate_all_intents(chunks: List[Dict], 
                         generate_for_items: bool = True,
                         generate_comparisons: bool = True) -> List[Intent]:
    """生成所有意圖（優化版）"""
    all_intents = []
    intent_id_counter = [1]
    validator = IntentQualityValidator()
    
    total_chunks = len(chunks)
    low_quality_count = 0
    
    # 1. 生成條文和項目意圖
    for i, chunk in enumerate(chunks, 1):
        clause = chunk["clause"]
        print(f"🔄 處理 [{i}/{total_chunks}]: {clause['clause_no']} {clause['clause_title']}")
        
        # 條文級別意圖
        clause_intents = generate_clause_intents(chunk, intent_id_counter, validator)
        all_intents.extend(clause_intents)
        
        # 統計低質量 intent
        low_quality = [i for i in clause_intents if i.quality_score < 0.8]
        if low_quality:
            low_quality_count += len(low_quality)
            print(f"  ⚠️  {len(low_quality)} 個低質量 intent（已自動修正）")
        
        # 項目級別意圖
        if generate_for_items and clause.get("items"):
            for item in clause["items"]:
                item_intents = generate_item_intents(chunk, item, intent_id_counter, validator)
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
    print(f"   - 低質量（已修正）: {low_quality_count}")
    print(f"   - 平均質量分數: {sum(i.quality_score for i in all_intents) / len(all_intents):.2f}")
    
    # 統計
    query_types = {}
    quality_distribution = {"優秀(>0.9)": 0, "良好(0.8-0.9)": 0, "中等(<0.8)": 0}
    
    for intent in all_intents:
        qt = intent.query_type
        query_types[qt] = query_types.get(qt, 0) + 1
        
        if intent.quality_score > 0.9:
            quality_distribution["優秀(>0.9)"] += 1
        elif intent.quality_score >= 0.8:
            quality_distribution["良好(0.8-0.9)"] += 1
        else:
            quality_distribution["中等(<0.8)"] += 1
    
    print(f"\n📊 意圖類型分布:")
    for qt, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {qt}: {count}")
    
    print(f"\n📊 質量分布:")
    for level, count in quality_distribution.items():
        print(f"   - {level}: {count}")
    
    return all_intents


# ==================== 保存函數 ====================

def save_intents(intents: List[Intent], output_path: str):
    """保存意圖到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "metadata": {
            "total_intents": len(intents),
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "version": "3.0",
            "enhancements": [
                "Few-shot Learning（正反例）",
                "後處理驗證與自動修正",
                "質量評分機制",
                "更嚴格的語義標籤要求"
            ],
            "average_quality_score": sum(i.quality_score for i in intents) / len(intents)
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
            "version": "3.0"
        },
        "chunks": chunks
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存 chunks（含意圖）至 {output_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("🔄 開始生成意圖（優化版）...")
    print("改進: Few-shot Learning + 後處理驗證 + 質量評分\n")
    
    # 載入 chunks
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured.json")
    
    if not os.path.exists(chunks_path):
        print(f"❌ 找不到 chunks 文件: {chunks_path}")
        print("請先運行 chunk_generator.py")
        exit(1)
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data["chunks"]
    print(f"📥 已載入 {len(chunks)} 個 chunks")
    
    # 生成意圖
    intents = generate_all_intents(
        chunks, 
        generate_for_items=True,
        generate_comparisons=True
    )
    
    # 保存
    intents_path = os.path.join(INDEX_DIR, "intents.json")
    save_intents(intents, intents_path)
    
    chunks_with_intents_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
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
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   - {cat}: {count}")
    
    print("\n🏷️  語義標籤統計:")
    for tag, count in sorted(semantic_tags_count.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   - {tag}: {count}")
