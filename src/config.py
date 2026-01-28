import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Versioning
INDEX_VERSION = "v1.2"

# Data files
PDF_PATH = os.path.join(BASE_DIR, "data", "source.pdf")

# Index files (with index version)
INDEX_DIR = os.path.join(BASE_DIR, "index", INDEX_VERSION)
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"

# Chunking
MAX_CHARS = 500

# Keyword filter
KEYWORDS = ["理賠", "投保", "延誤", "取消", "保險", "旅遊"]
SIM_THRESHOLD = 0.65
TOP_K = 3
