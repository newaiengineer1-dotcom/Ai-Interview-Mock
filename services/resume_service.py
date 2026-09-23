import json
from services.groq_service import chat

SYSTEM = """You are an ATS engine + senior recruiter. Return STRICT JSON ONLY:
{
 "match_percent": <0-100>,
 "technical_skills": {"score": <1-10>, "notes": "..."},
 "projects": {"score": <1-10>, "notes": "..."},
 "impact": {"score": <1-10>, "notes": "..."},
 "ats_readiness": {"score": <1-10>, "notes": "..."},
 "role_alignment": {"score": <1-10>, "notes": "..."},
 "strong_matches": ["...", "..."],
 "missing_gaps": ["...", "..."],
 "rewrites": [{"before": "...", "after": "..."}],
 "skill_weights": [{"skill": "Python", "weight": 20}, {"skill": "Django", "weight": 20}],
 "plan": ["4-part interview plan bullet 1", "bullet 2", "bullet 3", "bullet 4"]
}
No markdown, no prose, ONLY JSON."""


def analyze_resume_jd(resume_text, jd_text, api_key=None):
    user = (
        f"RESUME:\n{resume_text[:4500]}\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:3500]}\n\n"
        f"Analyze."
    )
    try:
        raw = chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user}],
            api_key=api_key,
            temperature=0.3,
            max_tokens=1600,
            json_mode=True,
        )
        return json.loads(raw)
    except Exception:
        return {}
