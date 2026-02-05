"""
配置文件 - 集中管理系統配置
"""

import os

# ==================== 基礎目錄 ====================

# 專案根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# 索引文件目錄
INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

# PDF 文件資料夾
PDF_FOLDER = os.path.join(BASE_DIR, "data")
os.makedirs(PDF_FOLDER, exist_ok=True)

# PDF 文件完整路徑
PDF_PATH = os.path.join(PDF_FOLDER, "source.pdf")

# ==================== 模型配置 ====================

# 嵌入模型
EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_TYPE = "openai" 

# OpenAI Embedding 模型
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"  # 或 "text-embedding-3-small", "text-embedding-ada-002"

SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-zh-v1.5"

# OpenAI 模型
OPENAI_MODEL_INTENT = "gpt-4o-mini"  # 用於生成意圖
OPENAI_MODEL_ANSWER = "gpt-4o-mini"  # 用於生成答案

# 生成參數
TEMPERATURE_INTENT = 0.3  # 意圖生成溫度
TEMPERATURE_ANSWER = 0.1  # 答案生成溫度

# ==================== 檢索配置 ====================

# 默認檢索參數
DEFAULT_TOP_K_INTENTS = 5  # 檢索 top-k 意圖
DEFAULT_TOP_K_CLAUSES = 3  # 返回 top-k 條文

# ==================== LLM 配置 ====================

MAX_RETRIES = 3         # 最大重試次數
REQUEST_INTERVAL = 0.5  # 請求間隔（秒）

# ==================== 文件路徑 ====================

CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks_structured.json")
INTENTS_FILE = os.path.join(INDEX_DIR, "intents.json")
CHUNKS_WITH_INTENTS_FILE = os.path.join(INDEX_DIR, "chunks_structured_with_intents.json")

# ==================== 日誌配置 ====================

VERBOSE = True           # 是否啟用詳細日誌
LOG_LEVEL = "INFO"       # 日誌級別
