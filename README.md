# 🎯 InterviewAI — Intelligent Mock Interview Platform

Agentic, multi-modal AI interview coach — Streamlit + Groq (Llama-3.3-70B + Whisper) + SQLite + TF-IDF RAG.

## ✨ Features
- Categorized RAG over Resume & JD (Skills / Projects / Experience / Education)
- Human-like, short, contextual questions (real interviewer tone)
- 6 Interviewer Personalities × 6 Interview Types
- Adaptive difficulty state machine
- 6-dimensional evaluation + ideal answer + missed points
- ATS Resume Gap Analyzer with XYZ rewrites
- Voice mode with WPM / Fillers / Clarity
- Coding mode with Big-O analysis
- Executive Analytics Dashboard (Plotly)
- SQLite session history & progression
- Downloadable Markdown + Styled HTML reports

## 🚀 Run Locally
```bash
git clone <your-repo>
cd InterviewAI
pip install -r requirements.txt
export GROQ_API_KEY="your_key"      # or paste in sidebar
streamlit run app.py
