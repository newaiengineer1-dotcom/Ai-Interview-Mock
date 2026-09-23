## 🆕 v2.0 Updates (September 2026)

### Model Migration
- **`llama-3.3-70b-versatile` → `openai/gpt-oss-120b`**
  The old model was **shut down by Groq on 16 August 2026**.
  GPT-OSS-120B is the official replacement with:
  - 131K context window (vs 128K)
  - ~500 tps (vs ~200 tps)
  - 90.0% MMLU (vs ~82%)
  - 62.4% SWE-Bench (vs ~45%)

### New: Suggested Best Answer
When your answer scores below 7/10, the AI now generates a **short, humanized, CV-specific answer** — exactly what a strong candidate would say in a real interview. No generic templates.

### New: Anti-Hallucination Guardrails
- **Stage 1 (deterministic)** : Checks that technical terms in the answer actually appear in the retrieved resume/JD context.
- **Stage 2 (LLM judge)** : GPT-OSS-120B verifies every factual claim against context with temperature=0.
- Flagged answers show a warning badge with unsupported claims listed.
