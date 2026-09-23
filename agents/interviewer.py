from services.groq_service import chat

PERSONALITIES = {
    "Friendly & Encouraging": "warm, supportive; gently probes deeper",
    "Professional & Structured": "calm senior hiring manager; structured and calibrated",
    "Strict & Demanding": "skeptical; stresses production reality; challenges buzzwords",
    "FAANG-Style": "distributed systems, Big-O trade-offs, scale and concurrency",
    "Startup CTO": "pragmatic; velocity vs tech debt; execution-focused",
    "HR Manager": "culture fit, teamwork, conflict resolution, motivation",
}

INTERVIEW_TYPES = [
    "Technical & System Design",
    "Behavioral (STAR)",
    "HR & Culture Fit",
    "System Design",
    "Coding & Algorithms",
    "Mixed Technical & Behavioral",
]

SYSTEM_PROMPT = """You are a senior interviewer conducting a real interview.
STRICT RULES:
- Ask exactly ONE short, natural, conversational question (max 2 sentences).
- The question MUST reference a specific project, skill, responsibility, or requirement from the RESUME/JD context.
- Sound like a real human interviewer on a call — casual, sharp, no filler.
- Never say "Great question", never add preamble, never use bullet points, never explain.
- Do NOT repeat questions already asked.
- Output ONLY the question text. Nothing else."""

def build_context_block(resume_ctx, jd_ctx):
    parts = []
    if resume_ctx:
        parts.append("RESUME CONTEXT:\n" + "\n".join(
            f"- [{c['category']}] {c['text'][:280]}" for c in resume_ctx))
    if jd_ctx:
        parts.append("JD CONTEXT:\n" + "\n".join(
            f"- [{c['category']}] {c['text'][:280]}" for c in jd_ctx))
    return "\n\n".join(parts) or "(no context provided)"

def generate_question(role, interview_type, personality, difficulty,
                      resume_ctx, jd_ctx, history, api_key=None):
    persona = PERSONALITIES.get(personality, "professional")
    ctx = build_context_block(resume_ctx, jd_ctx)
    hist = "\n".join(
        f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer'][:220]}"
        for i, h in enumerate(history[-5:])
    ) or "(none yet)"

    user = f"""Role: {role}
Interview type: {interview_type}
Interviewer style: {personality} ({persona})
Target difficulty: {difficulty}

{ctx}

Previous Q&A:
{hist}

Ask ONE short question (max 2 sentences) appropriate for a {interview_type} interview at {difficulty} difficulty, tightly tied to the context above."""
    q = chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        api_key=api_key, temperature=0.75, max_tokens=110,
    ).strip().strip('"').strip()
    return q.split("\n")[0].strip()
