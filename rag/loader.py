import io
import re

RESUME_SECTIONS = {
    "skills": ["skills", "technical skills", "core skills", "technologies", "tech stack", "tools"],
    "experience": ["experience", "work experience", "professional experience", "employment", "internship"],
    "projects": ["projects", "personal projects", "selected projects", "key projects", "portfolio"],
    "education": ["education", "academics", "qualifications", "degree", "certifications"],
}

JD_SECTIONS = {
    "required_skills": ["required skills", "requirements", "must have", "must-have", "skills required"],
    "responsibilities": ["responsibilities", "what you'll do", "what you will do", "role", "duties", "day to day"],
    "qualifications": ["qualifications", "preferred", "nice to have", "education", "experience required"],
}

def _read_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:
        return ""

def _read_docx(data: bytes) -> str:
    try:
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

def read_file(uploaded) -> str:
    data = uploaded.read()
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        return _read_pdf(data)
    if name.endswith(".docx"):
        return _read_docx(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def split_sections(text: str, section_map: dict) -> dict:
    if not text:
        return {k: "" for k in section_map} | {"_other": ""}
    lines = text.replace("\r", "").split("\n")
    result = {k: [] for k in section_map}
    result["_other"] = []
    current = "_other"
    for line in lines:
        low = line.strip().lower().rstrip(":").strip()
        matched = None
        if 0 < len(low) < 60:
            for cat, keys in section_map.items():
                if any(low == k or low.startswith(k + " ") or low.startswith(k + ":") for k in keys):
                    matched = cat
                    break
        if matched:
            current = matched
            continue
        result[current].append(line)
    return {k: "\n".join(v).strip() for k, v in result.items()}
