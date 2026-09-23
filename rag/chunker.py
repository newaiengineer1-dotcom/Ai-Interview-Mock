import re

def chunk_text(text: str, max_chars: int = 420):
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks, buf = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) > 10]

def build_index(section_texts: dict):
    docs = []
    for cat, text in (section_texts or {}).items():
        if not text or cat == "_other":
            continue
        for ch in chunk_text(text):
            docs.append({"category": cat, "text": ch})
    return docs
