"""
Answer Generator - 基於檢索結果生成最終答案

✅ 修復：相似度顯示問題（優先使用 rerank_score）

功能：
1. 從檢索引擎獲取相關條文
2. 構建結構化提示詞
3. 調用 LLM 生成專業答案
4. 支持多輪對話（可選）
"""

import json
import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from core.retrieval_engine import RetrievalEngine
from config import INDEX_DIR


# ==================== 初始化 ====================

load_dotenv()
client = OpenAI()


# ==================== 提示詞模板 ====================

SYSTEM_PROMPT = """你是專業的旅遊保險條款問答專家。你的任務是根據提供的保險條文，準確、清晰地回答使用者的問題。

回答要求：
1. **準確性**: 嚴格基於提供的條文內容，不要編造信息
2. **結構化**: 使用清晰的段落和條列式說明
3. **引用來源**: 明確標註引用的條文編號和項目
4. **完整性**: 包含適用條件、例外情況、注意事項等
5. **易讀性**: 使用簡單明瞭的語言，避免過度專業術語

回答格式建議：
- 先給出簡短的直接答案
- 然後詳細說明條件和細節
- 最後補充例外情況或注意事項
- 每個要點都標註來源條文

語氣：專業、友善、耐心"""


USER_PROMPT_TEMPLATE = """請根據以下保險條文回答使用者的問題。

【相關條文】
{context}

【使用者問題】
{query}

請提供詳細且結構化的回答。"""


# ==================== 答案生成器 ====================

class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, 
                 retrieval_engine: RetrievalEngine,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.1):
        """
        初始化答案生成器
        
        Args:
            retrieval_engine: 檢索引擎實例
            model: OpenAI 模型名稱
            temperature: 生成溫度
        """
        self.retrieval_engine = retrieval_engine
        self.model = model
        self.temperature = temperature
    
    def generate(self,
                query: str,
                top_k_intents: int = 5,
                top_k_clauses: int = 3,
                include_sources: bool = True) -> Dict:
        """
        生成答案
        
        Args:
            query: 用戶查詢
            top_k_intents: 檢索前 K 個意圖
            top_k_clauses: 使用前 K 個條文
            include_sources: 是否在響應中包含來源信息
        
        Returns:
            包含答案和元數據的字典
        """
        # 檢索相關條文
        retrieval_result = self.retrieval_engine.retrieve(
            query,
            top_k_intents=top_k_intents,
            top_k_clauses=top_k_clauses,
            include_metadata=True
        )
        
        # 構建上下文
        context = self._format_context(retrieval_result["top_clauses"])
        
        # 構建提示詞
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        
        # 調用 LLM
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # 構建結果
            result = {
                "query": query,
                "answer": answer,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            if include_sources:
                result["sources"] = self._extract_sources(retrieval_result["top_clauses"])
                result["top_intents"] = [
                    {
                        "intent_id": intent["intent_id"],
                        "user_query": intent["intent_data"]["user_query"],
                        "category": intent["intent_data"]["category"],
                        # ✅ 修復：優先使用 hybrid_score
                        "similarity": intent.get("hybrid_score", intent.get("similarity_score", 0))
                    }
                    for intent in retrieval_result["top_intents"]
                ]
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "answer": f"抱歉，生成答案時發生錯誤: {str(e)}",
                "error": str(e)
            }
    
    def _format_context(self, clauses: List[Dict]) -> str:
        """
        格式化條文為上下文
        
        Args:
            clauses: 條文列表
        
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        for i, clause in enumerate(clauses, 1):
            # 來源類型標籤
            source_label = {
                "mother": "【母條文】",
                "item": "【子項目】",
                "subitem": "【子款項】",
                "referenced": "【引用條文】"
            }.get(clause["source_type"], "【其他】")
            
            # 位置信息
            location = clause["clause_id"]
            if clause.get("item_no"):
                location += f" 第{clause['item_no']}項"
            if clause.get("subitem_no"):
                location += f" ({clause['subitem_no']})"
            
            # 組合
            context_parts.append(
                f"{source_label} {location}\n"
                f"{clause['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, clauses: List[Dict]) -> List[Dict]:
        """
        提取來源信息
        
        ✅ 修復：優先使用 rerank_score，fallback 到 similarity_score
        
        Args:
            clauses: 條文列表
        
        Returns:
            來源信息列表
        """
        sources = []
        
        for clause in clauses:
            # ✅ 關鍵修復：優先讀取 rerank_score
            score = clause.get("rerank_score", clause.get("similarity_score", 0.0))
            
            source = {
                "clause_id": clause["clause_id"],
                "source_type": clause["source_type"],
                "similarity_score": score  # 保持字段名稱一致
            }
            
            if clause.get("item_no"):
                source["item_no"] = clause["item_no"]
            if clause.get("subitem_no"):
                source["subitem_no"] = clause["subitem_no"]
            
            sources.append(source)
        
        return sources


# ==================== 對話式生成器（可選） ====================

class ConversationalAnswerGenerator(AnswerGenerator):
    """支持多輪對話的答案生成器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    def generate_with_history(self,
                              query: str,
                              **kwargs) -> Dict:
        """
        基於歷史對話生成答案
        
        Args:
            query: 用戶查詢
            **kwargs: 其他參數
        
        Returns:
            包含答案和元數據的字典
        """
        # 檢索相關條文
        retrieval_result = self.retrieval_engine.retrieve(
            query,
            top_k_intents=kwargs.get('top_k_intents', 5),
            top_k_clauses=kwargs.get('top_k_clauses', 3),
            include_metadata=True
        )
        
        # 構建上下文
        context = self._format_context(retrieval_result["top_clauses"])
        
        # 構建消息列表
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # 添加歷史對話
        messages.extend(self.conversation_history)
        
        # 添加當前查詢
        current_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        messages.append({"role": "user", "content": current_prompt})
        
        # 調用 LLM
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # 更新歷史
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # 保持歷史長度（最多保留最近 10 輪）
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                "query": query,
                "answer": answer,
                "model": self.model,
                "conversation_length": len(self.conversation_history) // 2,
                "sources": self._extract_sources(retrieval_result["top_clauses"]),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "query": query,
                "answer": f"抱歉，生成答案時發生錯誤: {str(e)}",
                "error": str(e)
            }
    
    def reset_history(self):
        """重置對話歷史"""
        self.conversation_history = []


# ==================== 主程序（測試） ====================

if __name__ == "__main__":
    # 初始化檢索引擎
    intents_path = os.path.join(INDEX_DIR, "intents.json")
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
    
    if not os.path.exists(intents_path) or not os.path.exists(chunks_path):
        print("❌ 請先運行 chunk_generator.py 和 intent_generator.py")
        exit(1)
    
    retrieval_engine = RetrievalEngine(
        intents_path, 
        chunks_path,
        use_bm25=True,
        use_cross_encoder=True
    )
    
    # 初始化答案生成器
    answer_gen = AnswerGenerator(retrieval_engine)
    
    # 測試查詢
    test_queries = [
        "什麼情況下可以申請旅遊延誤賠償？",
        "行李遺失後應該如何申請理賠？",
        "哪些原因屬於不可理賠範圍？",
        "班機延誤多久可以理賠？",
    ]
    
    print("\n" + "="*80)
    print("🤖 旅遊保險問答系統測試")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n【問題 {i}】{query}")
        print("-" * 80)
        
        result = answer_gen.generate(
            query,
            top_k_intents=5,
            top_k_clauses=3,
            include_sources=True
        )
        
        print(f"\n{result['answer']}")
        
        if 'sources' in result:
            print("\n📚 參考條文:")
            for source in result['sources']:
                location = source['clause_id']
                if source.get('item_no'):
                    location += f" 第{source['item_no']}項"
                print(f"  - {location} (相似度: {source['similarity_score']:.3f})")
        
        if 'usage' in result:
            print(f"\n💡 Token 使用: {result['usage']['total_tokens']} "
                  f"(prompt: {result['usage']['prompt_tokens']}, "
                  f"completion: {result['usage']['completion_tokens']})")
        
        print("=" * 80)