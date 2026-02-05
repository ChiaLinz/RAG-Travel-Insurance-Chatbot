"""
Retrieval Engine - RAG 檢索引擎

檢索流程：
1. Query Embedding - 用戶問題嵌入
2. Intent Retrieval - 檢索 Top-N 意圖
3. Clause Expansion - 擴展相關條文（母條文 + 子項目 + 被引用條文）
4. Reranking - 使用語義相似度重排序
5. Context Building - 構建最終上下文
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from config import INDEX_DIR, EMBEDDING_TYPE, OPENAI_EMBEDDING_MODEL, SENTENCE_TRANSFORMER_MODEL


# ==================== 嵌入模型初始化 ====================

class EmbeddingModel:
    """統一的嵌入模型接口"""
    
    def __init__(self):
        self.model_type = EMBEDDING_TYPE
        
        if self.model_type == "openai":
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()
            self.client = OpenAI()
            self.model_name = OPENAI_EMBEDDING_MODEL
            print(f"🔄 使用 OpenAI Embedding: {self.model_name}")
            
        elif self.model_type == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            self.model_name = SENTENCE_TRANSFORMER_MODEL
            print(f"🔄 使用 Sentence Transformer: {self.model_name}")
            
        else:
            raise ValueError(f"不支持的 embedding 類型: {self.model_type}")
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        將文本編碼為向量
        
        Args:
            texts: 文本列表
            show_progress: 是否顯示進度條
        
        Returns:
            嵌入向量數組 (n_texts, embedding_dim)
        """
        if self.model_type == "openai":
            # OpenAI API 批量嵌入
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        elif self.model_type == "sentence-transformers":
            # Sentence Transformer
            return self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
    
    def cosine_similarity(self, query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        """
        計算餘弦相似度
        
        Args:
            query_emb: 查詢向量 (1, embedding_dim)
            corpus_embs: 語料庫向量 (n_corpus, embedding_dim)
        
        Returns:
            相似度分數 (n_corpus,)
        """
        # 確保是 2D 數組
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        # 計算餘弦相似度
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        corpus_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        similarities = np.dot(query_norm, corpus_norm.T)[0]
        
        return similarities


# 全局嵌入模型實例
EMBED_MODEL = EmbeddingModel()


# ==================== 數據結構 ====================

@dataclass
class RetrievalResult:
    """檢索結果"""
    intent_id: str
    intent_data: Dict
    similarity_score: float
    
    def to_dict(self):
        return {
            "intent_id": self.intent_id,
            "intent_data": self.intent_data,
            "similarity_score": self.similarity_score
        }


@dataclass
class ExpandedClause:
    """擴展的條文"""
    source_type: str  # "mother", "item", "subitem", "referenced"
    clause_id: str
    item_no: Optional[str]
    subitem_no: Optional[str]
    content: str
    raw_text: str
    similarity_score: float = 0.0
    
    def to_dict(self):
        return {
            "source_type": self.source_type,
            "clause_id": self.clause_id,
            "item_no": self.item_no,
            "subitem_no": self.subitem_no,
            "content": self.content,
            "raw_text": self.raw_text,
            "similarity_score": self.similarity_score
        }


# ==================== 嵌入索引 ====================

class IntentIndex:
    """意圖嵌入索引"""
    
    def __init__(self, intents: List[Dict]):
        """
        初始化意圖索引
        
        Args:
            intents: 意圖列表
        """
        self.intents = intents
        self.intent_map = {intent["intent_id"]: intent for intent in intents}
        
        # 構建檢索語料（包含更多上下文信息）
        self.corpus = []
        for intent in intents:
            # 組合多個字段以提高檢索質量
            parts = [
                f"問題: {intent['user_query']}",
                f"內容: {intent['excerpt']}",
            ]
            
            if intent.get("conditions"):
                parts.append(f"條件: {'; '.join(intent['conditions'])}")
            
            if intent.get("category"):
                parts.append(f"類別: {intent['category']}")
            
            corpus_text = " | ".join(parts)
            self.corpus.append(corpus_text)
        
        # 生成嵌入
        print("🔄 正在生成意圖嵌入...")
        self.embeddings = EMBED_MODEL.encode(self.corpus, show_progress=True)
        print(f"✅ 已生成 {len(self.corpus)} 個意圖的嵌入 (維度: {self.embeddings.shape[1]})")
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        檢索最相關的意圖
        
        Args:
            query: 用戶查詢
            top_k: 返回前 K 個結果
        
        Returns:
            RetrievalResult 列表
        """
        # 查詢嵌入
        query_embedding = EMBED_MODEL.encode([query], show_progress=False)
        
        # 計算相似度
        similarities = EMBED_MODEL.cosine_similarity(query_embedding, self.embeddings)
        
        # 獲取 top-k 索引
        top_indices = similarities.argsort()[::-1][:top_k]
        
        # 構建結果
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                intent_id=self.intents[idx]["intent_id"],
                intent_data=self.intents[idx],
                similarity_score=float(similarities[idx])
            ))
        
        return results


# ==================== 條文擴展 ====================

class ClauseExpander:
    """條文擴展器"""
    
    def __init__(self, chunks: List[Dict]):
        """
        初始化條文擴展器
        
        Args:
            chunks: 條文 chunks 列表
        """
        # 建立快速查找映射
        self.clause_map = {}  # clause_id -> chunk
        self.item_map = {}     # (clause_id, item_no) -> item
        self.subitem_map = {}  # (clause_id, item_no, subitem_no) -> subitem
        
        for chunk in chunks:
            clause = chunk["clause"]
            clause_id = clause["clause_id"]
            
            # 條文級別映射
            self.clause_map[clause_id] = chunk
            
            # 項目級別映射
            for item in clause.get("items", []):
                item_key = (clause_id, item["item_no"])
                self.item_map[item_key] = item
                
                # 款項級別映射
                for subitem in item.get("sub_items", []):
                    subitem_key = (clause_id, item["item_no"], subitem["subitem_no"])
                    self.subitem_map[subitem_key] = subitem
    
    def expand_from_intent(self, intent: Dict) -> List[ExpandedClause]:
        """
        根據意圖擴展相關條文
        
        Args:
            intent: 意圖數據
        
        Returns:
            ExpandedClause 列表
        """
        expanded = []
        clause_id = intent["clause_id"]
        item_no = intent.get("item_no")
        subitem_no = intent.get("subitem_no")
        
        # 1. 母條文（總是包含）
        if clause_id in self.clause_map:
            chunk = self.clause_map[clause_id]
            clause = chunk["clause"]
            
            expanded.append(ExpandedClause(
                source_type="mother",
                clause_id=clause_id,
                item_no=None,
                subitem_no=None,
                content=clause["context"],
                raw_text=clause["raw_text"]
            ))
        
        # 2. 特定項目（如果意圖針對某個項目）
        if item_no:
            item_key = (clause_id, item_no)
            if item_key in self.item_map:
                item = self.item_map[item_key]
                
                expanded.append(ExpandedClause(
                    source_type="item",
                    clause_id=clause_id,
                    item_no=item_no,
                    subitem_no=None,
                    content=item["context"],
                    raw_text=item["raw_text"]
                ))
        
        # 3. 特定款項（如果意圖針對某個款項）
        if item_no and subitem_no:
            subitem_key = (clause_id, item_no, subitem_no)
            if subitem_key in self.subitem_map:
                subitem = self.subitem_map[subitem_key]
                
                expanded.append(ExpandedClause(
                    source_type="subitem",
                    clause_id=clause_id,
                    item_no=item_no,
                    subitem_no=subitem_no,
                    content=subitem["context"],
                    raw_text=subitem["raw_text"]
                ))
        
        # 4. 被引用的條文
        for ref_clause_id in intent.get("referenced_clauses", []):
            if ref_clause_id in self.clause_map:
                ref_chunk = self.clause_map[ref_clause_id]
                ref_clause = ref_chunk["clause"]
                
                expanded.append(ExpandedClause(
                    source_type="referenced",
                    clause_id=ref_clause_id,
                    item_no=None,
                    subitem_no=None,
                    content=ref_clause["context"],
                    raw_text=ref_clause["raw_text"]
                ))
        
        return expanded


# ==================== 重排序 ====================

class SemanticReranker:
    """語義重排序器"""
    
    @staticmethod
    def rerank(query: str, 
               clauses: List[ExpandedClause], 
               top_k: int = 3) -> List[ExpandedClause]:
        """
        使用語義相似度重排序條文
        
        Args:
            query: 用戶查詢
            clauses: 候選條文列表
            top_k: 返回前 K 個結果
        
        Returns:
            重排序後的 ExpandedClause 列表
        """
        if not clauses:
            return []
        
        # 提取文本
        texts = [clause.raw_text for clause in clauses]
        
        # 計算嵌入
        query_emb = EMBED_MODEL.encode([query], show_progress=False)
        clause_embs = EMBED_MODEL.encode(texts, show_progress=False)
        
        # 計算相似度
        similarities = EMBED_MODEL.cosine_similarity(query_emb, clause_embs)
        
        # 更新相似度分數
        for i, clause in enumerate(clauses):
            clause.similarity_score = float(similarities[i])
        
        # 排序並返回 top-k
        sorted_clauses = sorted(clauses, key=lambda x: x.similarity_score, reverse=True)
        return sorted_clauses[:top_k]


# ==================== 檢索引擎 ====================

class RetrievalEngine:
    """RAG 檢索引擎"""
    
    def __init__(self, intents_path: str, chunks_path: str):
        """
        初始化檢索引擎
        
        Args:
            intents_path: 意圖 JSON 文件路徑
            chunks_path: Chunks JSON 文件路徑
        """
        # 載入數據
        print("📥 載入意圖數據...")
        with open(intents_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f)
        self.intents = intents_data["intents"]
        
        print("📥 載入條文數據...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = chunks_data["chunks"]
        
        # 初始化組件
        self.intent_index = IntentIndex(self.intents)
        self.clause_expander = ClauseExpander(self.chunks)
        self.reranker = SemanticReranker()
        
        print("✅ 檢索引擎初始化完成")
    
    def retrieve(self,
                query: str,
                top_k_intents: int = 5,
                top_k_clauses: int = 3,
                include_metadata: bool = True) -> Dict:
        """
        檢索相關條文
        
        Args:
            query: 用戶查詢
            top_k_intents: 檢索前 K 個意圖
            top_k_clauses: 返回前 K 個條文
            include_metadata: 是否包含元數據
        
        Returns:
            檢索結果字典
        """
        # Step 1: 意圖檢索
        intent_results = self.intent_index.search(query, top_k=top_k_intents)
        
        # Step 2: 條文擴展
        candidate_clauses = []
        for intent_result in intent_results:
            expanded = self.clause_expander.expand_from_intent(intent_result.intent_data)
            candidate_clauses.extend(expanded)
        
        # 去重（基於 clause_id + item_no + subitem_no）
        seen = set()
        unique_clauses = []
        for clause in candidate_clauses:
            key = (clause.clause_id, clause.item_no, clause.subitem_no)
            if key not in seen:
                seen.add(key)
                unique_clauses.append(clause)
        
        # Step 3: 重排序
        top_clauses = self.reranker.rerank(query, unique_clauses, top_k=top_k_clauses)
        
        # 構建結果
        result = {
            "query": query,
            "top_intents": [r.to_dict() for r in intent_results] if include_metadata else None,
            "top_clauses": [c.to_dict() for c in top_clauses]
        }
        
        return result
    
    def get_context_for_llm(self, query: str, **kwargs) -> str:
        """
        獲取用於 LLM 的格式化上下文
        
        Args:
            query: 用戶查詢
            **kwargs: 傳遞給 retrieve 的其他參數
        
        Returns:
            格式化的上下文字符串
        """
        result = self.retrieve(query, **kwargs)
        
        context_parts = []
        for i, clause in enumerate(result["top_clauses"], 1):
            source_label = {
                "mother": "母條文",
                "item": "子項目",
                "subitem": "子款項",
                "referenced": "引用條文"
            }.get(clause["source_type"], "其他")
            
            location = clause["clause_id"]
            if clause["item_no"]:
                location += f" 第{clause['item_no']}項"
            if clause["subitem_no"]:
                location += f" 第{clause['subitem_no']}款"
            
            context_parts.append(
                f"【條文 {i}】{source_label} - {location}\n"
                f"內容: {clause['content']}\n"
                f"相似度: {clause['similarity_score']:.3f}\n"
            )
        
        return "\n".join(context_parts)


# ==================== 主程序（測試） ====================

if __name__ == "__main__":
    # 初始化檢索引擎
    intents_path = os.path.join(INDEX_DIR, "intents_v2.json")
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents_v2.json")
    
    if not os.path.exists(intents_path):
        print(f"❌ 找不到意圖文件: {intents_path}")
        print("請先運行 intent_generator.py")
        exit(1)
    
    if not os.path.exists(chunks_path):
        print(f"❌ 找不到 chunks 文件: {chunks_path}")
        print("請先運行 chunk_generator.py 和 intent_generator.py")
        exit(1)
    
    engine = RetrievalEngine(intents_path, chunks_path)
    
    # 測試查詢
    test_queries = [
        "什麼情況下可以申請旅遊延誤賠償？",
        "行李遺失後應該如何申請理賠？",
        "哪些原因屬於不可理賠範圍？",
        "班機延誤多久可以理賠？"
    ]
    
    print("\n" + "="*60)
    print("🧪 測試檢索引擎")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 查詢: {query}")
        print("-" * 60)
        
        context = engine.get_context_for_llm(query, top_k_intents=5, top_k_clauses=3)
        print(context)
        print("-" * 60)