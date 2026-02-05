"""
Retrieval Engine - 混合檢索引擎

新增功能：
1. BM25 + Semantic 混合檢索
2. Cross-Encoder Reranking
3. 語義標籤過濾
4. 動態 Top-K 調整
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from config import INDEX_DIR, EMBEDDING_TYPE, OPENAI_EMBEDDING_MODEL, SENTENCE_TRANSFORMER_MODEL
from transformers import logging 
logging.set_verbosity_error()



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
        """將文本編碼為向量"""
        if self.model_type == "openai":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        elif self.model_type == "sentence-transformers":
            return self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
    
    def cosine_similarity(self, query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        """計算餘弦相似度"""
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        corpus_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        similarities = np.dot(query_norm, corpus_norm.T)[0]
        
        return similarities


# 全局嵌入模型實例
EMBED_MODEL = EmbeddingModel()


# ==================== BM25 檢索器 ====================

class BM25Retriever:
    """BM25 關鍵詞檢索器"""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25
        
        Args:
            corpus: 文本語料庫
            k1: BM25 參數（控制詞頻飽和度）
            b: BM25 參數（控制文檔長度歸一化）
        """
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            self.jieba = jieba
            self.tokenize = lambda x: list(jieba.cut(x))
            
            # 分詞
            tokenized_corpus = [self.tokenize(doc) for doc in corpus]
            
            # 構建 BM25
            self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
            self.corpus = corpus
            
            print(f"✅ BM25 索引構建完成（{len(corpus)} 文檔）")
            
        except ImportError:
            print("⚠️  未安裝 rank_bm25，BM25 檢索將不可用")
            print("   安裝: pip install rank-bm25 jieba")
            self.bm25 = None
    
    def search(self, query: str, top_k: int = 5) -> np.ndarray:
        """
        BM25 檢索
        
        Args:
            query: 查詢文本
            top_k: 返回前 K 個結果
        
        Returns:
            BM25 分數數組
        """
        if self.bm25 is None:
            # 如果 BM25 不可用，返回零分數
            return np.zeros(len(self.corpus))
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        return scores


# ==================== Cross-Encoder Reranker ====================

class CrossEncoderReranker:
    """Cross-Encoder 重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        初始化 Cross-Encoder
        
        Args:
            model_name: 模型名稱
        """
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"🔄 載入 Cross-Encoder: {model_name}")
            self.model = CrossEncoder(model_name)
            self.enabled = True
            print(f"✅ Cross-Encoder 載入完成")
            
        except ImportError:
            print("⚠️  未安裝 sentence-transformers，Cross-Encoder 將不可用")
            print("   安裝: pip install sentence-transformers")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  Cross-Encoder 載入失敗: {e}")
            self.enabled = False
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> Tuple[List[int], List[float]]:
        """
        重排序文檔
        
        Args:
            query: 查詢文本
            documents: 候選文檔列表
            top_k: 返回前 K 個結果
        
        Returns:
            (索引列表, 分數列表)
        """
        if not self.enabled or not documents:
            # 如果不可用，返回原始順序
            return list(range(min(top_k, len(documents)))), [1.0] * min(top_k, len(documents))
        
        # 構建 query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # 預測分數
        scores = self.model.predict(pairs)
        
        # 排序
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_scores = [scores[i] for i in ranked_indices]
        
        return ranked_indices.tolist(), ranked_scores


# ==================== 數據結構 ====================

@dataclass
class RetrievalResult:
    """檢索結果"""
    intent_id: str
    intent_data: Dict
    similarity_score: float
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    
    def to_dict(self):
        return {
            "intent_id": self.intent_id,
            "intent_data": self.intent_data,
            "similarity_score": self.similarity_score,
            "bm25_score": self.bm25_score,
            "hybrid_score": self.hybrid_score
        }


@dataclass
class ExpandedClause:
    """擴展的條文"""
    source_type: str
    clause_id: str
    item_no: Optional[str]
    subitem_no: Optional[str]
    content: str
    raw_text: str
    similarity_score: float = 0.0
    rerank_score: float = 0.0
    
    def to_dict(self):
        return {
            "source_type": self.source_type,
            "clause_id": self.clause_id,
            "item_no": self.item_no,
            "subitem_no": self.subitem_no,
            "content": self.content,
            "raw_text": self.raw_text,
            "similarity_score": self.similarity_score,
            "rerank_score": self.rerank_score
        }


# ==================== 意圖索引（混合檢索）====================

class IntentIndex:
    """意圖嵌入索引（混合檢索版）"""
    
    def __init__(self, intents: List[Dict], use_bm25: bool = True):
        """
        初始化意圖索引
        
        Args:
            intents: 意圖列表
            use_bm25: 是否啟用 BM25
        """
        self.intents = intents
        self.intent_map = {intent["intent_id"]: intent for intent in intents}
        
        # 構建檢索語料
        self.corpus = []
        for intent in intents:
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
        
        # 1. 語義嵌入
        print("🔄 正在生成意圖嵌入（Semantic）...")
        self.embeddings = EMBED_MODEL.encode(self.corpus, show_progress=True)
        print(f"✅ 已生成 {len(self.corpus)} 個意圖的嵌入 (維度: {self.embeddings.shape[1]})")
        
        # 2. BM25 索引
        self.bm25_retriever = None
        if use_bm25:
            print("🔄 構建 BM25 索引...")
            self.bm25_retriever = BM25Retriever(self.corpus)
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               semantic_weight: float = 0.85,
               bm25_weight: float = 0.15,
               semantic_tags_filter: Optional[List[str]] = None) -> List[RetrievalResult]:
        """
        混合檢索（增強版）
        
        Args:
            query: 用戶查詢
            top_k: 返回前 K 個結果
            semantic_weight: 語義檢索權重
            bm25_weight: BM25 權重
            semantic_tags_filter: 語義標籤過濾（可選）
        
        Returns:
            RetrievalResult 列表
        """
        # 1. 語義檢索
        query_embedding = EMBED_MODEL.encode([query], show_progress=False)
        semantic_scores = EMBED_MODEL.cosine_similarity(query_embedding, self.embeddings)
        
        # 2. BM25 檢索
        if self.bm25_retriever is not None:
            bm25_scores = self.bm25_retriever.search(query, top_k=top_k)
            # 歸一化 BM25 分數
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
        else:
            bm25_scores = np.zeros_like(semantic_scores)
        
        # 3. 混合分數
        hybrid_scores = semantic_weight * semantic_scores + bm25_weight * bm25_scores
        
        # 4. 語義標籤過濾
        if semantic_tags_filter:
            for i, intent in enumerate(self.intents):
                intent_tags = intent.get("semantic_tags", [])
                # 如果沒有匹配的標籤，降低分數
                if not any(tag in intent_tags for tag in semantic_tags_filter):
                    hybrid_scores[i] *= 0.5  # 降低 50%
        
        # 5. 獲取 top-k
        top_indices = hybrid_scores.argsort()[::-1][:top_k]
        
        # 6. 構建結果
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                intent_id=self.intents[idx]["intent_id"],
                intent_data=self.intents[idx],
                similarity_score=float(semantic_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
                hybrid_score=float(hybrid_scores[idx])
            ))
        
        return results


# ==================== 條文擴展器 ====================

class ClauseExpander:
    """條文擴展器"""
    
    def __init__(self, chunks: List[Dict]):
        """初始化條文擴展器"""
        self.clause_map = {}
        self.item_map = {}
        self.subitem_map = {}
        
        for chunk in chunks:
            clause = chunk["clause"]
            clause_id = clause["clause_id"]
            
            self.clause_map[clause_id] = chunk
            
            for item in clause.get("items", []):
                item_key = (clause_id, item["item_no"])
                self.item_map[item_key] = item
                
                for subitem in item.get("sub_items", []):
                    subitem_key = (clause_id, item["item_no"], subitem["subitem_no"])
                    self.subitem_map[subitem_key] = subitem
    
    def expand_from_intent(self, intent: Dict) -> List[ExpandedClause]:
        """根據意圖擴展相關條文"""
        expanded = []
        clause_id = intent["clause_id"]
        item_no = intent.get("item_no")
        subitem_no = intent.get("subitem_no")
        
        # 母條文
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
        
        # 特定項目
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
        
        # 特定款項
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
        
        # 被引用的條文
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


# ==================== 檢索引擎（混合版）====================

class RetrievalEngine:
    """RAG 檢索引擎（混合檢索 + Cross-Encoder）"""
    
    def __init__(self, 
                 intents_path: str, 
                 chunks_path: str,
                 use_bm25: bool = True,
                 use_cross_encoder: bool = True):
        """
        初始化檢索引擎
        
        Args:
            intents_path: 意圖 JSON 文件路徑
            chunks_path: Chunks JSON 文件路徑
            use_bm25: 是否啟用 BM25
            use_cross_encoder: 是否啟用 Cross-Encoder
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
        self.intent_index = IntentIndex(self.intents, use_bm25=use_bm25)
        self.clause_expander = ClauseExpander(self.chunks)
        
        # Cross-Encoder
        self.use_cross_encoder = use_cross_encoder
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
        else:
            self.cross_encoder = None
        
        print("✅ 檢索引擎 初始化完成")
    
    def _detect_semantic_tags(self, query: str) -> List[str]:
        """檢測查詢中的語義標籤"""
        tags = []
        
        # 動作詞映射
        action_map = {
            "遺失": ["遺失", "丟", "弄丟", "不見", "失竊"],
            "延誤": ["延誤", "晚點", "delay", "誤點"],
            "損失": ["損失", "損壞", "毀損"],
            "取消": ["取消", "中止"],
            "更改": ["更改", "變更", "改變"],
            "竊盜": ["偷", "竊", "搶"],
        }
        
        for tag, keywords in action_map.items():
            if any(kw in query for kw in keywords):
                tags.append(tag)
        
        # 特殊標籤
        if any(kw in query for kw in ["不", "哪些", "除外"]):
            tags.append("不保")
        
        return tags
    
    def _smart_top_k(self, query: str) -> Dict[str, int]:
        """動態調整 top_k"""
        # 複數問題 → 增加召回
        if any(kw in query for kw in ["哪些", "所有", "全部", "什麼"]):
            return {"intents": 10, "clauses": 5}
        
        # 簡單問題 → 減少召回
        elif any(kw in query for kw in ["多久", "幾小時", "幾天"]):
            return {"intents": 3, "clauses": 2}
        
        # 默認
        else:
            return {"intents": 5, "clauses": 3}
    
    def retrieve(self,
                query: str,
                top_k_intents: Optional[int] = None,
                top_k_clauses: Optional[int] = None,
                include_metadata: bool = True,
                auto_adjust_topk: bool = True) -> Dict:
        """
        檢索相關條文（增強版）
        
        Args:
            query: 用戶查詢
            top_k_intents: 檢索前 K 個意圖（None = 自動）
            top_k_clauses: 返回前 K 個條文（None = 自動）
            include_metadata: 是否包含元數據
            auto_adjust_topk: 是否自動調整 top_k
        
        Returns:
            檢索結果字典
        """
        # 動態調整 top_k
        if auto_adjust_topk:
            smart_k = self._smart_top_k(query)
            top_k_intents = top_k_intents or smart_k["intents"]
            top_k_clauses = top_k_clauses or smart_k["clauses"]
        else:
            top_k_intents = top_k_intents or 5
            top_k_clauses = top_k_clauses or 3
        
        # 檢測語義標籤
        semantic_tags = self._detect_semantic_tags(query)
        
        # Stage 1: 意圖檢索（混合檢索）
        intent_results = self.intent_index.search(
            query, 
            top_k=top_k_intents,
            semantic_tags_filter=semantic_tags if semantic_tags else None
        )
        
        # Stage 2: 條文擴展
        candidate_clauses = []
        for intent_result in intent_results:
            expanded = self.clause_expander.expand_from_intent(intent_result.intent_data)
            candidate_clauses.extend(expanded)
        
        # 去重
        seen = set()
        unique_clauses = []
        for clause in candidate_clauses:
            key = (clause.clause_id, clause.item_no, clause.subitem_no)
            if key not in seen:
                seen.add(key)
                unique_clauses.append(clause)
        
        # Stage 3: Cross-Encoder 重排序
        if self.use_cross_encoder and self.cross_encoder and self.cross_encoder.enabled:
            # 提取文本
            texts = [clause.raw_text for clause in unique_clauses]
            
            # 重排序
            ranked_indices, rerank_scores = self.cross_encoder.rerank(
                query, texts, top_k=top_k_clauses
            )
            
            # 更新分數並選擇 top-k
            top_clauses = []
            for idx, score in zip(ranked_indices, rerank_scores):
                clause = unique_clauses[idx]
                clause.rerank_score = score
                top_clauses.append(clause)
        else:
            # 語義重排序
            query_emb = EMBED_MODEL.encode([query], show_progress=False)
            clause_texts = [clause.raw_text for clause in unique_clauses]
            clause_embs = EMBED_MODEL.encode(clause_texts, show_progress=False)
            
            similarities = EMBED_MODEL.cosine_similarity(query_emb, clause_embs)
            
            for i, clause in enumerate(unique_clauses):
                clause.similarity_score = float(similarities[i])
                clause.rerank_score = float(similarities[i])
            
            sorted_clauses = sorted(unique_clauses, key=lambda x: x.similarity_score, reverse=True)
            top_clauses = sorted_clauses[:top_k_clauses]
        
        # 構建結果
        result = {
            "query": query,
            "detected_semantic_tags": semantic_tags,
            "top_k_intents": top_k_intents,
            "top_k_clauses": top_k_clauses,
            "top_intents": [r.to_dict() for r in intent_results] if include_metadata else None,
            "top_clauses": [c.to_dict() for c in top_clauses]
        }
        
        return result
    
    def get_context_for_llm(self, query: str, **kwargs) -> str:
        """獲取用於 LLM 的格式化上下文"""
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
            
            # 顯示 rerank 分數
            score = clause.get("rerank_score", clause.get("similarity_score", 0))
            
            context_parts.append(
                f"【條文 {i}】{source_label} - {location}\n"
                f"內容: {clause['content']}\n"
                f"相關度: {score:.3f}\n"
            )
        
        return "\n".join(context_parts)


# ==================== 主程序（測試）====================

if __name__ == "__main__":
    # 初始化檢索引擎
    intents_path = os.path.join(INDEX_DIR, "intents.json")
    chunks_path = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")
    
    if not os.path.exists(intents_path):
        print(f"❌ 找不到意圖文件: {intents_path}")
        print("請先運行 intent_generator.py")
        exit(1)
    
    if not os.path.exists(chunks_path):
        print(f"❌ 找不到chunks 文件: {chunks_path}")
        print("請先運行 intent_generator.py")
        exit(1)
    
    engine = RetrievalEngine(
        intents_path, 
        chunks_path,
        use_bm25=True,
        use_cross_encoder=True
    )
    
    # 測試查詢
    test_queries = [
        "什麼情況下可以申請旅遊延誤賠償？",
        "行李遺失後應該如何申請理賠？",
        "哪些原因屬於不可理賠範圍？",
        "班機延誤多久可以理賠？"
    ]
    
    print("\n" + "="*60)
    print("🧪 測試檢索引擎（混合檢索 + Cross-Encoder）")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 查詢: {query}")
        print("-" * 60)
        
        result = engine.retrieve(query, include_metadata=True)
        
        print(f"檢測到的語義標籤: {result['detected_semantic_tags']}")
        print(f"Top-K 配置: intents={result['top_k_intents']}, clauses={result['top_k_clauses']}")
        print()
        
        # 顯示 top intents
        print("Top Intents:")
        for i, intent in enumerate(result["top_intents"][:3], 1):
            print(f"  {i}. {intent['intent_data']['user_query']}")
            print(f"     Semantic: {intent['similarity_score']:.3f} | "
                  f"BM25: {intent['bm25_score']:.3f} | "
                  f"Hybrid: {intent['hybrid_score']:.3f}")
        
        print()
        print("Top Clauses:")
        for i, clause in enumerate(result["top_clauses"], 1):
            print(f"  {i}. {clause['clause_id']}")
            print(f"     Rerank Score: {clause['rerank_score']:.3f}")
        
        print("-" * 60)
