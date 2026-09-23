import json
from services.groq_service import chat

DIMENSIONS = [
    "Technical Accuracy",
    "Completeness",
    "Depth & Nuance",
    "Communication & Conciseness",
    "Problem Solving & Reasoning",
    "Role Relevance",
]

# UPDATED: Now includes suggested_best_answer and hallucination_risk
EVAL_SYSTEM = """You are a senior interview evaluator. Return STRICT JSON ONLY:
{
  "scores": {
    "Technical Accuracy": <1-10>,
    "Completeness": <1-10>,
    "Depth & Nuance": <1-10>,
    "Communication & Conciseness": <1-10>,
    "Problem Solving & Reasoning": <1-10>,
    "Role Relevance": <1-10>
  },
  "overall": <float 1-10>,
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "ideal_answer": "concise ideal answer, 3-6 sentences",
  "missed": ["specific missing point 1", "specific missing point 2"],
  "verdict": "one short sentence",
  "suggested_best_answer": "IF the candidate's answer scored below 7 overall, provide a SHORT (2-3 sentences max), natural, conversational answer that a strong candidate would actually say out loud in an interview. It must be humanized, direct, reference the specific project/skill mentioned in the resume context, and avoid buzzwords. If the answer scored 7+, set this to empty string.",
  "hallucination_risk": {
    "flagged": <true/false>,
    "reason": "one-line explanation if flagged, else empty string",
    "unsupported_claims": ["claim1", "claim2"]
  }
}
No prose, no markdown fences, ONLY valid JSON."""

# Anti-hallucination system prompt for the judge
GROUNDEDNESS_SYSTEM = """You are a strict fact-checking judge for RAG systems.
You receive: (1) CONTEXT chunks retrieved from a resume/JD, (2) a QUESTION, (3) an ANSWER.
Your job: determine if every factual claim in the ANSWER is supported by the CONTEXT.

Return STRICT JSON ONLY:
{
  "grounded": <true|false>,
  "confidence": <0.0-1.0>,
  "unsupported_claims": ["exact claim text that is NOT in context", ...],
  "reason": "one-line explanation"
}

Rules:
- If the answer contains specific numbers, technologies, or project names NOT present in the context → grounded=false.
- If the answer is generic and adds no new facts → grounded=true (safe).
- If the answer contradicts the context → grounded=false.
- Be strict. When in doubt, flag it."""

def _ctx_block(resume_ctx, jd_ctx):
    parts = []
    if resume_ctx:
        parts.append("RESUME:\n" + "\n".join(c["text"] for c in resume_ctx[:6]))
    if jd_ctx:
        parts.append("JD:\n" + "\n".join(c["text"] for c in jd_ctx[:6]))
    return "\n\n".join(parts)

def evaluate_answer(question, answer, role, resume_ctx, jd_ctx, api_key=None):
    ctx = _ctx_block(resume_ctx, jd_ctx)
    user = f"""Role: {role}
{ctx}

Question: {question}
Candidate Answer: {answer}

Evaluate strictly. Return only JSON."""
    try:
        raw = chat(
            [{"role": "system", "content": EVAL_SYSTEM},
             {"role": "user", "content": user}],
            api_key=api_key, temperature=0.2, max_tokens=900, json_mode=True,
            reasoning_effort="low",
        )
        data = json.loads(raw)
    except Exception:
        data = {}
    s = data.get("scores", {}) or {}
    for d in DIMENSIONS:
        try:
            s[d] = float(s.get(d, 5))
        except Exception:
            s[d] = 5.0
    data["scores"] = s
    try:
        data["overall"] = float(data.get("overall", sum(s.values()) / len(DIMENSIONS)))
    except Exception:
        data["overall"] = sum(s.values()) / len(DIMENSIONS)
    data.setdefault("strengths", [])
    data.setdefault("gaps", [])
    data.setdefault("ideal_answer", "")
    data.setdefault("missed", [])
    data.setdefault("verdict", "")
    data.setdefault("suggested_best_answer", "")
    data.setdefault("hallucination_risk", {"flagged": False, "reason": "", "unsupported_claims": []})
    return data

# ---- Anti-Hallucination: Groundedness Check ----
def check_groundedness(question, answer, context_chunks, api_key=None):
    """
    Two-stage groundedness gate:
    Stage 1: Deterministic — check if answer shares key terms with context.
    Stage 2: LLM judge — verify factual claims.
    Returns dict: {grounded, confidence, unsupported_claims, reason}
    """
    if not answer or not context_chunks:
        return {"grounded": True, "confidence": 1.0,
                "unsupported_claims": [], "reason": "No context or answer to check."}

    # Stage 1: Deterministic keyword overlap check (fast, free)
    context_text = " ".join(c.get("text", "") for c in context_chunks).lower()
    answer_lower = answer.lower()
    # Extract technical-looking tokens (words with caps, digits, or length > 5)
    import re
    tokens = set(re.findall(r"\b[A-Za-z][A-Za-z0-9+#\.\-]{4,}\b", answer))
    tech_tokens = {t for t in tokens if any(
        c.isupper() for c in t[1:]) or t.lower() in [
        "python", "django", "flask", "docker", "kubernetes", "react",
        "node", "sql", "mongodb", "postgres", "redis", "aws", "azure",
        "gcp", "api", "rest", "graphql", "microservices", "ci", "cd",
        "git", "linux", "java", "javascript", "typescript", "go", "rust",
    ]}
    if tech_tokens:
        overlap = sum(1 for t in tech_tokens if t.lower() in context_text)
        overlap_ratio = overlap / len(tech_tokens)
        if overlap_ratio < 0.3:
            return {
                "grounded": False,
                "confidence": 0.85,
                "unsupported_claims": list(tech_tokens)[:5],
                "reason": f"Answer mentions {len(tech_tokens)} technical terms but only {overlap} appear in context."
            }

    # Stage 2: LLM judge (only if stage 1 passes — save tokens)
    ctx_str = "\n\n".join(f"[{i+1}] {c.get('text','')[:300]}" for i, c in enumerate(context_chunks[:5]))
    user = f"""CONTEXT:
{ctx_str}

QUESTION: {question}
ANSWER: {answer}

Judge groundedness. Return JSON only."""
    try:
        raw = chat(
            [{"role": "system", "content": GROUNDEDNESS_SYSTEM},
             {"role": "user", "content": user}],
            api_key=api_key, temperature=0.0, max_tokens=300, json_mode=True,
            reasoning_effort="low",
        )
        return json.loads(raw)
    except Exception:
        return {"grounded": True, "confidence": 0.5,
                "unsupported_claims": [], "reason": "Judge unavailable; defaulting to grounded."}

CODE_SYSTEM = """You are a senior engineer reviewing a coding interview answer.
Return STRICT JSON ONLY:
{
  "correctness": {"score": <1-10>, "notes": "..."},
  "time_complexity": "Big-O + one-line why",
  "space_complexity": "Big-O + one-line why",
  "code_quality": {"score": <1-10>, "notes": "..."},
  "edge_cases": ["...", "..."],
  "follow_up": "one follow-up optimization challenge",
  "overall": <1-10>,
  "verdict": "one short sentence"
}"""

def evaluate_code(problem_title, problem_prompt, code, language="python", api_key=None):
    user = f"Problem: {problem_title}\n{problem_prompt}\n\nLanguage: {language}\nCode:\n```{language}\n{code}\n```"
    try:
        raw = chat(
            [{"role": "system", "content": CODE_SYSTEM},
             {"role": "user", "content": user}],
            api_key=api_key, temperature=0.2, max_tokens=800, json_mode=True,
            reasoning_effort="medium",
        )
        return json.loads(raw)
    except Exception:
        return {
            "correctness": {"score": 0, "notes": "Eval failed"},
            "time_complexity": "n/a", "space_complexity": "n/a",
            "code_quality": {"score": 0, "notes": "n/a"},
            "edge_cases": [], "follow_up": "",
            "overall": 0, "verdict": "Evaluation failed",
        }
