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
  "verdict": "one short sentence"
}
No prose, no markdown fences, ONLY valid JSON."""

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
            api_key=api_key, temperature=0.2, max_tokens=700, json_mode=True,
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
    return data

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
            api_key=api_key, temperature=0.2, max_tokens=700, json_mode=True,
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
