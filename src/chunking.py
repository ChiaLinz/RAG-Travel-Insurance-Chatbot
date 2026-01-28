import re
import json
import os
from datetime import datetime
from config import MAX_CHARS

# ---------- Text Cleaning ----------
def clean_text(text):
    # Remove lines with only numbers (usually page numbers) and extra blank lines
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

# ---------- chunk_type Inference ----------
def infer_chunk_type(text):
    # Simple rule-based
    if any(k in text for k in ["不負賠償責任", "不包括", "不保"]):
        return "exception"
    if any(k in text for k in ["應備文件", "申請", "理賠", "索取"]):
        return "procedure"
    if any(k in text for k in ["係指", "定義", "指"]):
        return "definition"
    return "coverage"

# ---------- semantic_scope Inference ----------
KEYWORDS = {
    "flight": ["航班", "班機", "延誤", "取消"],
    "baggage": ["行李", "遺失", "毀損", "箱", "提箱"],
    "typhoon": ["颱風", "天災", "暴風"],
    "claim": ["理賠", "申請", "文件", "索取"],
}

def extract_semantic_scope(text):
    scope = []
    for k, kws in KEYWORDS.items():
        if any(kw in text for kw in kws):
            scope.append(k)
    return scope if scope else ["general"]

# ---------- chunk_text v1.0 ----------
def chunk_text(text, version="1.0"):
    text = clean_text(text)
    chunks = []

    # regex patterns
    chapter_pat = re.compile(r"^(第[一二三四五六七八九十]+章)\s*(.+?)\s*$", re.MULTILINE)
    clause_pat = re.compile(r"^(第[一二三四五六七八九十]+條)\s*(.+?)\s*$", re.MULTILINE)
    item_pat = re.compile(r"\n([一二三四五六七八九十]+)、")

    chapters = list(chapter_pat.finditer(text))
    for i, ch in enumerate(chapters):
        ch_start = ch.end()
        ch_end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
        ch_text = text[ch_start:ch_end]
        chapter_no, chapter_title = ch.group(1), ch.group(2)

        clauses = list(clause_pat.finditer(ch_text))
        for j, cl in enumerate(clauses):
            cl_start = cl.end()
            cl_end = clauses[j + 1].start() if j + 1 < len(clauses) else len(ch_text)
            clause_body = ch_text[cl_start:cl_end].strip()
            clause_no, clause_title = cl.group(1), cl.group(2)

            # Item detection
            items = list(item_pat.finditer(clause_body))
            if not items:
                # No items, chunk the whole clause
                if clause_body:
                    chunks.extend(split_long_chunk(clause_body, chapter_no, chapter_title, clause_no, clause_title, None, version))
                continue

            # Has items, extract context first
            context = clause_body[:items[0].start()].strip()
            for k, it in enumerate(items):
                it_start = it.end()
                it_end = items[k + 1].start() if k + 1 < len(items) else len(clause_body)
                item_text = clause_body[it_start:it_end].strip()
                combined_text = f"{context}\n{it.group(1)}、{item_text}" if context else f"{it.group(1)}、{item_text}"
                chunks.extend(split_long_chunk(combined_text, chapter_no, chapter_title, clause_no, clause_title,
                                               {"no": it.group(1), "text": item_text}, version))
    return chunks

# ---------- Split Long Chunk ----------
def split_long_chunk(text, chapter_no, chapter_title, clause_no, clause_title, item, version):
    # Return directly if smaller than MAX_CHARS
    if len(text) <= MAX_CHARS:
        return [create_chunk(text, chapter_no, chapter_title, clause_no, clause_title, item, version)]
    
    # Otherwise, split by punctuation
    sentences = re.split(r'(?<=[。；])', text)
    chunks = []
    buffer = ""
    for s in sentences:
        if len(buffer + s) <= MAX_CHARS:
            buffer += s
        else:
            if buffer.strip():
                chunks.append(create_chunk(buffer.strip(), chapter_no, chapter_title, clause_no, clause_title, item, version))
            buffer = s
    if buffer.strip():
        chunks.append(create_chunk(buffer.strip(), chapter_no, chapter_title, clause_no, clause_title, item, version))
    return chunks

# ---------- Create Chunk ----------
def create_chunk(text, chapter_no, chapter_title, clause_no, clause_title, item, version):
    return {
        "chapter": {"no": chapter_no, "title": chapter_title},
        "clause": {"no": clause_no, "title": clause_title},
        "item": item,
        "text": text,
        "version": version,
        "chunk_type": infer_chunk_type(text),
        "semantic_scope": extract_semantic_scope(text)
    }

# ---------- Save Chunks ----------
def save_chunks(chunks, version="v1.0", base_path="index", notes="Initial version"):
    dir_path = os.path.join(base_path, version)
    os.makedirs(dir_path, exist_ok=True)
    chunks_path = os.path.join(dir_path, "chunks.json")
    meta_path = os.path.join(dir_path, "metadata.json")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    metadata = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "num_chunks": len(chunks)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {chunks_path}")
    print(f"Metadata saved to {meta_path}")
