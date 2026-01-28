import os
import faiss
import json
import numpy as np
from datetime import datetime

from config import PDF_PATH, INDEX_DIR, EMBEDDING_MODEL
from embedding import get_embedding
from chunking import chunk_text
from load_pdf import load_pdf

def build_faiss_index(pdf_path=PDF_PATH, 
                      version="v1.0", 
                      embedding_model=EMBEDDING_MODEL,
                      notes="Initial version"):
    """
    Build FAISS index from PDF using structured chunking.
    Versioned: will create index and chunks under index/{version}/
    """

    # Versioned paths
    version_dir = os.path.join(INDEX_DIR, version)
    os.makedirs(version_dir, exist_ok=True)
    index_path = os.path.join(version_dir, "faiss.index")
    chunks_path = os.path.join(version_dir, "chunks.json")
    metadata_path = os.path.join(version_dir, "metadata.json")

    # 1. Load PDF
    print(f"[INFO] Loading PDF from {pdf_path} ...")
    text = load_pdf(pdf_path)

    # 2. Chunk PDF text
    print(f"[INFO] Chunking PDF (version={version}) ...")
    chunks = chunk_text(text, version=version)
    print(f"[INFO] {len(chunks)} chunks generated.")

    # 3. Compute embeddings
    print("[INFO] Generating embeddings ...")
    vectors = []
    for c in chunks:
        emb = get_embedding(c["text"], model=embedding_model)
        vectors.append(emb)
    vectors = np.array(vectors, dtype="float32")
    dim = vectors.shape[1]

    # 4. Build FAISS index
    print(f"[INFO] Building FAISS index with dimension {dim} ...")
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)
    print(f"[INFO] FAISS index saved to {index_path}")

    # 5. Save chunks with metadata
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Chunks saved to {chunks_path}")

    # 6. Save metadata for version tracking
    metadata = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "num_chunks": len(chunks),
        "embedding_model": embedding_model
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Metadata saved to {metadata_path}")

    print("[INFO] Index build completed successfully.")


if __name__ == "__main__":
    # Example usage
    build_faiss_index(version="v1.0", notes="Initial version")
