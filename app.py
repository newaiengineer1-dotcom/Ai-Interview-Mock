import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))import streamlit as st import pandas as pd import plotly.express as px
import plotly.graph_objects as go from database.db import init_db, create_session, add_turn, update_session, get_sessions, get_turns, delete_session, delete_all_sessions, get_session_stats
from services.groq_service import chat, transcribe, DEFAULT_MODEL
from rag.loader import read_file, split_sections, RESUME_SECTIONS, JD_SECTIONS
from rag.chunker import build_index
from rag.retriever import CategoryRetriever
from agents.interviewer import generate_question, PERSONALITIES, INTERVIEW_TYPES
from agents.evaluator import evaluate_answer, evaluate_code, check_groundedness, DIMENSIONS
from agents.manager import InterviewManager, DIFFICULTY_LEVELS
from services.resume_service import analyze_resume_jd
from services.report_service import build_markdown_report, build_html_report
from utils.helpers import analyze_speech
from utils.coding_problems import PROBLEMS

init_db()

st.set_page_config(
    page_title="InterviewAI Studio",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
  background: radial-gradient(1000px 500px at 12% -10%, rgba(34,211,238,0.10), transparent 60%),
              radial-gradient(900px 480px at 90% -5%, rgba(139,92,246,0.10), transparent 60%),
              #0a0e17;
}
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, li, span, label, div { color: #cbd5e1; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.cockpit-top {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(90deg, #0f172a 0%, #0b1220 100%);
  border: 1px solid rgba(34,211,238,0.18);
  border-radius: 14px; padding: 10px 16px; margin-bottom: 14px;
}
.cockpit-left { display: flex; align-items: center; gap: 10px; }
.cockpit-logo {
  width: 34px; height: 34px; border-radius: 10px;
  background: linear-gradient(135deg, #22d3ee, #8b5cf6);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; color: #0a0e17; font-size: 15px;
}
.cockpit-title { font-weight: 700; color: #f1f5f9; font-size: 15px; }
.cockpit-sub { font-size: 11px; color: #64748b; letter-spacing: 0.14em; text-transform: uppercase; }
.cockpit-right { display: flex; gap: 8px; align-items: center; }
.chip {
  background: rgba(34,211,238,0.10); color: #67e8f9;
  border: 1px solid rgba(34,211,238,0.25);
  padding: 4px 10px; border-radius: 999px; font-size: 11px;
}
.chip.ok { background: rgba(16,185,129,0.10); color: #6ee7b7; border-color: rgba(16,185,129,0.30); }
.qhead {
  display: flex; align-items: center; justify-content: space-between;
  background: rgba(15,23,42,0.7); border: 1px solid rgba(34,211,238,0.15);
  border-radius: 12px; padding: 10px 14px; margin-bottom: 12px;
}
.qhead-title { font-weight: 600; color: #e2e8f0; font-size: 14px; }
.qhead-meta { display: flex; gap: 8px; align-items: center; }
.pill-diff {
  background: linear-gradient(90deg, #ef4444, #f59e0b);
  color: #0a0e17; padding: 3px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 700;
}
.pill-soft {
  background: rgba(148,163,184,0.10); color: #cbd5e1;
  padding: 3px 10px; border-radius: 999px; font-size: 11px;
  border: 1px solid rgba(148,163,184,0.18);
}
.panel {
  background: linear-gradient(180deg, rgba(15,23,42,0.72), rgba(11,18,32,0.72));
  border: 1px solid rgba(34,211,238,0.14);
  border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
}
.panel-title {
  font-size: 11px; color: #67e8f9; letter-spacing: 0.14em;
  text-transform: uppercase; margin-bottom: 10px; font-weight: 600;
}
.panel-sub { font-size: 12px; color: #94a3b8; }
.ai-row { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
.ai-avatar {
  width: 44px; height: 44px; border-radius: 12px;
  background: conic-gradient(from 210deg, #22d3ee, #8b5cf6, #ec4899, #22d3ee);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; color: #0a0e17; font-size: 15px;
}
.ai-name { font-weight: 700; color: #f1f5f9; font-size: 14px; }
.ai-role { font-size: 11px; color: #94a3b8; }
.live-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #22d3ee;
  display: inline-block; margin-right: 6px;
}
.qtext { font-size: 15px; line-height: 1.55; color: #e2e8f0; }
.mrow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }
.mtile {
  background: rgba(15,23,42,0.6);
  border: 1px solid rgba(34,211,238,0.16);
  border-radius: 12px; padding: 12px 10px; text-align: center;
}
.mtile .v { font-size: 1.35rem; font-weight: 800; color: #67e8f9; }
.mtile .l { font-size: 10px; color: #94a3b8; text-transform: uppercase; margin-top: 5px; }
.mtile.ok .v { color: #6ee7b7; }
.mic-wrap { text-align: center; padding: 14px 0 6px 0; }
.mic-circle {
  width: 74px; height: 74px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #22d3ee, #0891b2 70%);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 20px; color: #0a0e17; font-weight: 700;
}
.mic-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; margin-top: 10px; }
button[data-baseweb="tab"] {
  background: transparent !important; color: #94a3b8 !important;
  font-weight: 600 !important; font-size: 13px !important;
  border-radius: 10px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #67e8f9 !important; background: rgba(34,211,238,0.08) !important;
}
div[data-baseweb="tab-highlight"] { background-color: #22d3ee !important; }
.stButton > button {
  background: linear-gradient(135deg, #22d3ee, #8b5cf6);
  color: #0a0e17; border: 0; border-radius: 10px;
  font-weight: 700; padding: 0.5rem 1rem;
}
section[data-testid="stSidebar"] {
  background: #080c15; border-right: 1px solid rgba(34,211,238,0.12);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
  color: #67e8f9 !important; font-size: 12px !important;
  letter-spacing: 0.14em; text-transform: uppercase;
}
textarea, input {
  background: rgba(15,23,42,0.75) !important;
  border: 1px solid rgba(34,211,238,0.18) !important;
  color: #e2e8f0 !important;
}
.chip-ok, .chip-gap, .tag {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 11px; margin: 3px 4px 3px 0;
}
.chip-ok { background: rgba(16,185,129,0.15); color: #6ee7b7; }
.chip-gap { background: rgba(239,68,68,0.15); color: #fca5a5; }
.tag { background: rgba(34,211,238,0.10); color: #67e8f9; border: 1px solid rgba(34,211,238,0.25); }
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 20px 0; }
.kpi {
  background: linear-gradient(145deg, rgba(15,23,42,0.85), rgba(11,18,32,0.85));
  border: 1px solid rgba(34,211,238,0.18);
  border-radius: 16px; padding: 18px; position: relative; overflow: hidden;
}
.kpi::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #22d3ee, #8b5cf6);
}
.kpi .icon {
  width: 36px; height: 36px; border-radius: 10px;
  background: rgba(34,211,238,0.10);
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; margin-bottom: 10px;
  border: 1px solid rgba(34,211,238,0.22);
  color: #67e8f9; font-weight: 700;
}
.kpi .val { font-size: 1.9rem; font-weight: 800; color: #f1f5f9; line-height: 1; }
.kpi .lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; margin-top: 6px; font-weight: 600; }
.kpi .trend { font-size: 11px; margin-top: 8px; font-weight: 600; color: #94a3b8; }
.section-title {
  display: flex; align-items: center; gap: 10px;
  font-size: 15px; font-weight: 700; color: #f1f5f9;
  margin: 22px 0 12px 0;
}
.section-title::before {
  content: ''; width: 4px; height: 18px; border-radius: 2px;
  background: linear-gradient(180deg, #22d3ee, #8b5cf6);
}
.section-title .count {
  margin-left: auto; font-size: 11px; color: #67e8f9;
  background: rgba(34,211,238,0.10); padding: 3px 10px;
  border-radius: 999px; border: 1px solid rgba(34,211,238,0.25);
  font-weight: 600;
}
.session-row {
  display: flex; align-items: center;
  background: rgba(15,23,42,0.6);
  border: 1px solid rgba(148,163,184,0.10);
  border-radius: 12px; padding: 14px 18px;
}
.session-row .sid { font-size: 12px; color: #67e8f9; font-weight: 700; min-width: 60px; }
.session-row .meta { flex: 1; margin-left: 18px; }
.session-row .role { font-weight: 600; color: #e2e8f0; font-size: 13px; }
.session-row .sub { font-size: 11px; color: #64748b; margin-top: 2px; }
.score-hi { color: #6ee7b7; background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.30); }
.score-mid { color: #fde047; background: rgba(250,204,21,0.10); border: 1px solid rgba(250,204,21,0.30); }
.score-lo { color: #fca5a5; background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.30); }
.empty {
  text-align: center; padding: 46px 20px;
  border: 1px dashed rgba(148,163,184,0.20);
  border-radius: 14px; color: #64748b;
}
.empty .ico { font-size: 32px; margin-bottom: 10px; opacity: 0.6; font-weight: 700; }
.empty .h { font-size: 15px; color: #cbd5e1; font-weight: 600; }
.empty .p { font-size: 12px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


def _summary_scores(hist):
    if not hist:
        return {d: 0 for d in DIMENSIONS}
    agg = {d: 0.0 for d in DIMENSIONS}
    for h in hist:
        s = h.get("evaluation", {}).get("scores", {})
        for d in DIMENSIONS:
            try:
                agg[d] += float(s.get(d, 0))
            except (TypeError, ValueError):
                pass
    n = len(hist)
    return {d: agg[d] / n for d in DIMENSIONS}


def _radar_fig(scores):
    cats = list(DIMENSIONS)
    vals = [float(scores.get(c, 0)) for c in cats]
    if not cats:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", line=dict(color="#22d3ee", width=2),
        fillcolor="rgba(34,211,238,0.18)", name="Score"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor="rgba(148,163,184,0.20)",
                            tickfont=dict(color="#94a3b8", size=9)),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=10)),
        ),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=360)
    return fig


def _init_state():
    defaults = {
        "api_key": "", "manager": None, "session_id": None,
        "history": [], "current_question": None,
        "resume_text": "", "jd_text": "",
        "resume_sections": {}, "jd_sections": {},
        "retriever": None, "role": "Backend Software Engineer",
        "itype": INTERVIEW_TYPES[0], "personality": "FAANG-Style",
        "difficulty": "Medium", "analyzed": False, "ats_report": None,
        "speech_metrics": None, "coding_result": None,
        "_last_transcript": "", "_last_groundedness": None,
        "candidate_name": "Alex Chen", "_view_session": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _get_api_key():
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.environ.get("GROQ_API_KEY", "") or st.session_state.get("api_key", "")


def _safe_text(value):
    import html
    return html.escape(str(value or ""))


_init_state()

q_now = len(st.session_state.history) + 1
total_q = 6
diff = st.session_state.manager.difficulty if st.session_state.manager else st.session_state.difficulty
progress_pct = int(min(100, (len(st.session_state.history) / total_q) * 100))

st.markdown(f"""
<div class="cockpit-top">
  <div class="cockpit-left">
    <div class="cockpit-logo">AI</div>
    <div>
      <div class="cockpit-title">InterviewAI</div>
      <div class="cockpit-sub">Studio Cockpit</div>
    </div>
  </div>
  <div class="cockpit-right">
    <span class="chip ok">Ready</span>
    <span class="chip">Model: {_safe_text(DEFAULT_MODEL)}</span>
    <span class="chip">Difficulty: {_safe_text(diff)}</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Groq API")
    loaded_key = _get_api_key()
    if not loaded_key:
        st.session_state.api_key = st.text_input(
            "API Key", type="password",
            value=st.session_state.api_key,
            placeholder="gsk_...", label_visibility="collapsed")
    else:
        st.success("API key loaded")
    st.caption("Get a key at console.groq.com")
    st.caption(f"Model: {DEFAULT_MODEL}")

    st.markdown("---")
    st.markdown("### Candidate Setup")
    st.session_state.candidate_name = st.text_input("Name", value=st.session_state.candidate_name)
    st.session_state.role = st.text_input("Target Role", value=st.session_state.role)
    st.session_state.itype = st.selectbox("Interview Type", INTERVIEW_TYPES, index=INTERVIEW_TYPES.index(st.session_state.itype))
    st.session_state.personality = st.selectbox("Interviewer Persona", list(PERSONALITIES.keys()), index=list(PERSONALITIES.keys()).index(st.session_state.personality))
    st.session_state.difficulty = st.selectbox("Starting Difficulty", DIFFICULTY_LEVELS, index=DIFFICULTY_LEVELS.index(st.session_state.difficulty))

    st.markdown("---")
    st.markdown("### RAG and ATS")
    resume_file = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="resume_upload")
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="jd_upload")
    analyze_btn = st.button("Analyze Documents", use_container_width=True)

    st.markdown("---")
    if st.button("Reset Session", use_container_width=True):
        for k in ["manager", "session_id", "current_question", "ats_report", "speech_metrics", "coding_result", "_last_groundedness"]:
            st.session_state[k] = None
        st.session_state["history"] = []
        st.session_state["analyzed"] = False
        st.session_state["_last_transcript"] = ""
        st.rerun()

if analyze_btn:
    with st.spinner("Parsing and indexing..."):
        if resume_file:
            st.session_state.resume_text = read_file(resume_file)
            st.session_state.resume_sections = split_sections(st.session_state.resume_text, RESUME_SECTIONS)
        if jd_file:
            st.session_state.jd_text = read_file(jd_file)
            st.session_state.jd_sections = split_sections(st.session_state.jd_text, JD_SECTIONS)
        combined = {}
        for k, v in st.session_state.resume_sections.items():
            if v:
                combined[f"resume_{k}"] = v
        for k, v in st.session_state.jd_sections.items():
            if v:
                combined[f"jd_{k}"] = v
        if not combined:
            st.warning("Upload at least one Resume or Job Description.")
        else:
            docs = build_index(combined)
            st.session_state.retriever = CategoryRetriever(docs)
            st.session_state.analyzed = True
            st.success(f"Indexed {len(docs)} chunks.")

tabs = st.tabs(["1. Live Interview", "2. Evaluation", "3. Resume and ATS", "4. Coding IDE", "5. Dashboard", "6. History"])

with tabs[0]:
    st.markdown(f"""
    <div class="qhead">
      <div class="qhead-title">Question {q_now} of {total_q} - {_safe_text(st.session_state.itype)}</div>
      <div class="qhead-meta">
        <span class="pill-diff">{_safe_text(diff)}</span>
        <span class="pill-soft">{progress_pct}% complete</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.manager is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("Start Adaptive Session", use_container_width=True):
                api_key = _get_api_key()
                if not api_key:
                    st.error("Groq API key required. Add GROQ_API_KEY to secrets or enter in sidebar.")
                elif not st.session_state.analyzed and not st.session_state.resume_text and not st.session_state.jd_text:
                    st.warning("Upload Resume and/or JD first, then click Analyze.")
                else:
                    st.session_state.manager = InterviewManager(st.session_state.difficulty)
                    st.session_state.history = []
                    st.session_state.session_id = create_session(
                        st.session_state.role, st.session_state.itype,
                        st.session_state.personality, st.session_state.difficulty)
                    rt = st.session_state.retriever
                    r_ctx = rt.retrieve(st.session_state.role, k=4) if rt else []
                    j_ctx = [d for d in rt.docs if d["category"].startswith("jd_")][:3] if rt else []
                    with st.spinner("Preparing first question..."):
                        q = generate_question(
                            st.session_state.role, st.session_state.itype,
                            st.session_state.personality, st.session_state.difficulty,
                            r_ctx, j_ctx, [], api_key=api_key)
                    st.session_state.current_question = q
                    st.rerun()

    if st.session_state.manager and st.session_state.current_question:
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            st.markdown(f"""
            <div class="panel">
              <div class="ai-row">
                <div class="ai-avatar">AI</div>
                <div>
                  <div class="ai-name"><span class="live-dot"></span>Aria-6X</div>
                  <div class="ai-role">{_safe_text(st.session_state.personality)} - {_safe_text(diff)} tier</div>
                </div>
              </div>
              <div class="panel-title">Question</div>
              <div class="qtext">{_safe_text(st.session_state.current_question)}</div>
            </div>
            """, unsafe_allow_html=True)

            weak = st.session_state.manager.weak_areas[-3:] if st.session_state.manager else []
            if weak:
                chips = "".join(f'<span class="tag">{_safe_text(w)}</span>' for w in weak)
                st.markdown(f'<div class="panel"><div class="panel-title">Active Evaluation Focus</div>{chips}</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="panel">
              <div class="panel-title">Interviewer Tip</div>
              <div class="panel-sub">Answer in 30-60 seconds. Reference a specific project, the tech stack, and one measurable outcome.</div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            mode = st.radio("Answer Mode", ["Voice", "Text"], horizontal=True, key="answer_mode", label_visibility="collapsed")
            answer_text = ""

            if mode == "Voice":
                st.markdown("""
                <div class="panel">
                  <div class="panel-title">Live Neural Audio Buffer</div>
                  <div class="mic-wrap">
                    <div class="mic-circle">MIC</div>
                    <div class="mic-label">Click below to record</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                audio_data = None
                try:
                    audio_data = st.audio_input("Record", label_visibility="collapsed")
                except Exception:
                    st.info("Audio recording unavailable. Upload an audio file instead.")

                up = st.file_uploader("or upload audio", type=["wav", "mp3", "m4a", "webm"], label_visibility="collapsed", key="answer_audio_upload")
                if up is not None:
                    audio_data = up

                if audio_data is not None and st.button("Transcribe via Whisper"):
                    with st.spinner("Transcribing..."):
                        suffix = ".wav"
                        if hasattr(audio_data, "name"):
                            suffix = Path(audio_data.name).suffix or ".wav"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(audio_data.getvalue())
                            path = tmp.name
                        try:
                            text = transcribe(path, api_key=_get_api_key())
                            st.session_state["_last_transcript"] = text
                            st.session_state.speech_metrics = analyze_speech(text, 30)
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                        finally:
                            try:
                                os.unlink(path)
                            except OSError:
                                pass

                answer_text = st.text_area("Transcript", value=st.session_state.get("_last_transcript", ""), height=130, key="voice_answer", placeholder="Your transcript will appear here...")
            else:
                answer_text = st.text_area("Your Answer", height=260, key="typed_answer", placeholder="Type your answer - be specific and concise...")

            m = st.session_state.speech_metrics or {"wpm": 0, "filler_total": 0, "duration_sec": 0, "clarity": 0, "words": 0}
            st.markdown(f"""
            <div class="mrow">
              <div class="mtile"><div class="v">{m.get("wpm", 0)}</div><div class="l">Speaking Pace</div></div>
              <div class="mtile"><div class="v">{m.get("filler_total", 0)}</div><div class="l">Filler Words</div></div>
              <div class="mtile"><div class="v">{m.get("duration_sec", 0)}s</div><div class="l">Duration</div></div>
              <div class="mtile ok"><div class="v">{m.get("clarity", 0)}/10</div><div class="l">Clarity Score</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            b1, b2, b3 = st.columns([1, 1, 1.4])
            with b1:
                submit = st.button("Submit", use_container_width=True)
            with b2:
                skip = st.button("Skip", use_container_width=True)
            with b3:
                finish = st.button("Finish and Save", use_container_width=True)

            if submit or skip:
                if not answer_text and not skip:
                    st.warning("Provide an answer or skip.")
                else:
                    ans = answer_text if answer_text else "(skipped)"
                    rt = st.session_state.retriever
                    q = st.session_state.current_question
                    r_ctx = rt.retrieve(q + " " + ans[:200], k=4) if rt else []
                    j_ctx = [d for d in rt.docs if d["category"].startswith("jd_")][:3] if rt else []
                    with st.spinner("Evaluating..."):
                        ev = evaluate_answer(q, ans, st.session_state.role, r_ctx, j_ctx, api_key=_get_api_key())
                    with st.spinner("Groundedness check..."):
                        grounded = check_groundedness(q, ans, r_ctx + j_ctx, api_key=_get_api_key())
                    st.session_state["_last_groundedness"] = grounded
                    if not grounded.get("grounded", True):
                        ev["hallucination_risk"] = {"flagged": True, "reason": grounded.get("reason", ""), "unsupported_claims": grounded.get("unsupported_claims", [])}
                    st.session_state.manager.record(q, ev)
                    st.session_state.history.append({"question": q, "answer": ans, "evaluation": ev, "groundedness": grounded})
                    add_turn(st.session_state.session_id, len(st.session_state.history), q, ans, ev, ev.get("scores", {}))
                    with st.spinner("Next question..."):
                        next_q = generate_question(st.session_state.role, st.session_state.itype, st.session_state.personality, st.session_state.manager.difficulty, r_ctx, j_ctx, st.session_state.history, api_key=_get_api_key())
                    st.session_state.current_question = next_q
                    st.session_state["_last_transcript"] = ""
                    st.rerun()

            if finish and st.session_state.history:
                avg = sum(h["evaluation"].get("overall", 0) for h in st.session_state.history) / max(len(st.session_state.history), 1)
                update_session(st.session_state.session_id, avg, summary="Completed", metrics=st.session_state.speech_metrics or {})
                st.success(f"Session #{st.session_state.session_id} saved - avg {avg:.1f}/10")

        if st.session_state.history:
            st.markdown("---")
            st.markdown("### Feedback - Latest Answer")
            last = st.session_state.history[-1]
            ev = last["evaluation"]
            hal = ev.get("hallucination_risk", {})
            if hal.get("flagged"):
                st.error(f"Hallucination risk - {hal.get('reason', '')}")
            g = st.session_state.get("_last_groundedness")
            if g and not g.get("grounded", True):
                st.warning(f"Ungrounded - {g.get('reason', '')} ({g.get('confidence', 0):.0%})")

            f1, f2 = st.columns([1, 2])
            with f1:
                st.plotly_chart(_radar_fig(ev.get("scores", {})), use_container_width=True, key="live_radar")
            with f2:
                st.markdown(f"""
                <div class="panel">
                  <div class="panel-title">Overall Score</div>
                  <div style="font-size:2rem;font-weight:800;color:#67e8f9;">
                    {ev.get("overall", 0):.1f}<span style="font-size:1rem;color:#94a3b8;">/10</span>
                  </div>
                  <div class="panel-sub" style="margin-top:6px;">{_safe_text(ev.get("verdict", ""))}</div>
                </div>
                """, unsafe_allow_html=True)
                if ev.get("strengths"):
                    st.markdown("**Strengths**")
                    st.markdown("".join(f'<span class="chip-ok">{_safe_text(s)}</span>' for s in ev["strengths"]), unsafe_allow_html=True)
                if ev.get("gaps"):
                    st.markdown("**Gaps**")
                    st.markdown("".join(f'<span class="chip-gap">{_safe_text(s)}</span>' for s in ev["gaps"]), unsafe_allow_html=True)

            suggested = ev.get("suggested_best_answer", "").strip()
            if suggested:
                st.markdown(f"""
                <div class="panel" style="border-color:rgba(250,204,21,0.35);">
                  <div class="panel-title" style="color:#fde047;">Suggested Best Answer (Humanized)</div>
                  <div class="qtext">{_safe_text(suggested)}</div>
                </div>
                """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="panel"><div class="panel-title">Your Answer</div><div class="panel-sub">{_safe_text(last["answer"])}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="panel"><div class="panel-title">Ideal Model Answer</div><div class="panel-sub">{_safe_text(ev.get("ideal_answer") or "-")}</div></div>', unsafe_allow_html=True)
            with c2:
                missed = ev.get("missed", [])
                missed_html = "<br>".join("- " + _safe_text(x) for x in missed) if missed else "-"
                st.markdown(f'<div class="panel"><div class="panel-title">What Was Missed</div><div class="panel-sub">{missed_html}</div></div>', unsafe_allow_html=True)

            md = build_markdown_report(st.session_state.role, st.session_state.itype, st.session_state.personality, st.session_state.history, _summary_scores(st.session_state.history), st.session_state.speech_metrics)
            html = build_html_report(md)
            e1, e2 = st.columns(2)
            with e1:
                st.download_button("Download Markdown Report", md, file_name=f"report_{datetime.utcnow():%Y%m%d_%H%M}.md", mime="text/markdown", use_container_width=True)
            with e2:
                st.download_button("Download HTML Report", html, file_name=f"report_{datetime.utcnow():%Y%m%d_%H%M}.html", mime="text/html", use_container_width=True)

with tabs[1]:
    st.markdown("### 6-Dimensional Evaluation")
    hist = st.session_state.history
    if not hist:
        st.info("Complete at least one question first.")
    else:
        summary = _summary_scores(hist)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(_radar_fig(summary), use_container_width=True, key="eval_radar")
        with c2:
            df = pd.DataFrame({"Dimension": list(summary.keys()), "Score": list(summary.values())})
            fig = px.bar(df, x="Score", y="Dimension", orientation="h", color="Score", color_continuous_scale="Teal")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=360, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        for i, h in enumerate(hist, 1):
            ev = h["evaluation"]
            with st.expander(f"Q{i} - {h['question'][:80]} - {ev.get('overall', 0):.1f}/10"):
                st.markdown(f"**Your answer:** {h['answer']}")
                st.markdown(f"**Verdict:** {ev.get('verdict', '')}")
                if ev.get("ideal_answer"):
                    st.markdown(f"**Ideal:** {ev['ideal_answer']}")
                if ev.get("suggested_best_answer"):
                    st.markdown(f"**Suggested:** {ev['suggested_best_answer']}")

with tabs[2]:
    st.markdown("### Resume and ATS Gap Analyzer")
    if not st.session_state.resume_text or not st.session_state.jd_text:
        st.info("Upload Resume + JD in the sidebar, then click Analyze Documents.")
    else:
        if st.button("Run ATS Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                st.session_state.ats_report = analyze_resume_jd(st.session_state.resume_text, st.session_state.jd_text, api_key=_get_api_key())
        rep = st.session_state.ats_report
        if rep:
            k1, k2, k3, k4 = st.columns(4)
            items = [
                (k1, "Match", f'{rep.get("match_percent", 0)}%'),
                (k2, "Skills", rep.get("technical_skills", {}).get("score", 0)),
                (k3, "Projects", rep.get("projects", {}).get("score", 0)),
                (k4, "Impact", rep.get("impact", {}).get("score", 0)),
            ]
            for col, label, val in items:
                with col:
                    st.markdown(f'<div class="mtile"><div class="v">{_safe_text(val)}</div><div class="l">{_safe_text(label)}</div></div>', unsafe_allow_html=True)
            sw = rep.get("skill_weights", [])
            if sw:
                st.markdown("#### Skill Weighting (from JD)")
                df = pd.DataFrame(sw)
                fig = px.bar(df, x="weight", y="skill", orientation="h", color="weight", color_continuous_scale="Teal")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=320, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Strong Matches**")
                st.markdown("".join(f'<span class="chip-ok">{_safe_text(s)}</span>' for s in rep.get("strong_matches", [])), unsafe_allow_html=True)
            with c2:
                st.markdown("**Missing Gaps**")
                st.markdown("".join(f'<span class="chip-gap">{_safe_text(s)}</span>' for s in rep.get("missing_gaps", [])), unsafe_allow_html=True)
            for r in rep.get("rewrites", []):
                st.markdown(f'<div class="panel"><div class="panel-title">Rewrite</div><div class="panel-sub"><b>Before:</b> {_safe_text(r.get("before", ""))}</div><div class="panel-sub" style="margin-top:6px;"><b style="color:#6ee7b7;">After:</b> {_safe_text(r.get("after", ""))}</div></div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown("### Live Coding IDE")
    problem = st.selectbox("Problem", [f"{p['title']} - {p['difficulty']}" for p in PROBLEMS])
    title = problem.rsplit(" - ", 1)[0]
    prob = next(p for p in PROBLEMS if p["title"] == title)
    st.markdown(f'<div class="panel"><div class="panel-title">Problem - {_safe_text(prob["difficulty"])}</div><div class="qtext">{_safe_text(prob["prompt"])}</div><div class="panel-sub" style="margin-top:8px;"><b>Examples:</b> {_safe_text(prob["examples"])}</div></div>', unsafe_allow_html=True)
    lang = st.selectbox("Language", ["python", "javascript", "java", "cpp", "go"])
    code = st.text_area("Solution", height=260, placeholder="# write your solution here")
    if st.button("Evaluate Code", use_container_width=True):
        if not code.strip():
            st.warning("Write some code first.")
        else:
            with st.spinner("Analyzing..."):
                st.session_state.coding_result = evaluate_code(prob["title"], prob["prompt"], code, lang, api_key=_get_api_key())
    res = st.session_state.coding_result
    if res:
        c1, c2, c3 = st.columns(3)
        items = [
            (c1, "Correctness", res.get("correctness", {}).get("score", 0)),
            (c2, "Code Quality", res.get("code_quality", {}).get("score", 0)),
            (c3, "Overall", res.get("overall", 0)),
        ]
        for col, label, val in items:
            with col:
                st.markdown(f'<div class="mtile"><div class="v">{_safe_text(val)}</div><div class="l">{_safe_text(label)}</div></div>', unsafe_allow_html=True)
        st.markdown(f"**Time:** {res.get('time_complexity', '-')}")
        st.markdown(f"**Space:** {res.get('space_complexity', '-')}")
        if res.get("edge_cases"):
            st.markdown("**Edge cases:** " + ", ".join(str(x) for x in res["edge_cases"]))
        if res.get("follow_up"):
            st.info(f"**Follow-up:** {res['follow_up']}")
        if res.get("verdict"):
            st.success(res["verdict"])

with tabs[4]:
    hist = st.session_state.history
    sessions = get_sessions()
    stats = get_session_stats()
    total_q_now = len(hist)
    avg_now = sum(h["evaluation"].get("overall", 0) for h in hist) / total_q_now if total_q_now else 0
    best_q = max((h["evaluation"].get("overall", 0) for h in hist), default=0)
    grounded_ok = sum(1 for h in hist if h.get("groundedness", {}).get("grounded", True))
    grounded_pct = int((grounded_ok / total_q_now) * 100) if total_q_now else 0
    trend_html = '<span class="trend">no baseline</span>'
    if len(sessions) >= 2 and sessions[0]["avg_score"]:
        prev = sessions[1]["avg_score"] or 0
        cur = sessions[0]["avg_score"] or 0
        if prev > 0:
            delta = cur - prev
            if delta > 0.3:
                trend_html = f'<span class="trend" style="color:#6ee7b7;">+{delta:.1f} vs last</span>'
            elif delta < -0.3:
                trend_html = f'<span class="trend" style="color:#fca5a5;">{delta:.1f} vs last</span>'
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="icon">Q</div><div class="val">{total_q_now}</div><div class="lbl">Questions This Session</div>{trend_html}</div>
      <div class="kpi"><div class="icon">A</div><div class="val">{avg_now:.1f}</div><div class="lbl">Current Average</div><span class="trend">out of 10.0</span></div>
      <div class="kpi"><div class="icon">B</div><div class="val">{best_q:.1f}</div><div class="lbl">Best Answer</div><span class="trend">peak score</span></div>
      <div class="kpi"><div class="icon">G</div><div class="val">{grounded_pct}%</div><div class="lbl">Groundedness</div><span class="trend">anti-hallucination</span></div>
    </div>
    """, unsafe_allow_html=True)
    if not hist and not sessions:
        st.markdown('<div class="empty"><div class="ico">EMPTY</div><div class="h">No interview data yet</div><div class="p">Start an interview in Tab 1.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">Performance Overview <span class="count">LIVE</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            summary = _summary_scores(hist) if hist else {d: 0 for d in DIMENSIONS}
            st.plotly_chart(_radar_fig(summary), use_container_width=True, key="dash_radar")
        with c2:
            if hist:
                df = pd.DataFrame({"Q": [f"Q{i+1}" for i in range(len(hist))], "Score": [h["evaluation"].get("overall", 0) for h in hist]})
                fig = px.line(df, x="Q", y="Score", markers=True)
                fig.update_traces(line_color="#22d3ee", line_width=3, marker=dict(size=12, color="#8b5cf6"))
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", yaxis_range=[0, 10], height=360)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Complete a question to see the trajectory.")
        if stats["total"] > 0:
            st.markdown('<div class="section-title">All-Time Statistics <span class="count">SQLITE</span></div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            stat_items = [
                (s1, "S", "Total Sessions", stats["total"], "#67e8f9"),
                (s2, "A", "Avg Score", f'{stats["avg_score"]:.1f}', "#a5b4fc"),
                (s3, "B", "Best Session", f'{stats["best_score"]:.1f}', "#6ee7b7"),
                (s4, "L", "Lowest Session", f'{stats["worst_score"]:.1f}', "#fca5a5"),
            ]
            for col, icon, label, val, color in stat_items:
                with col:
                    st.markdown(f'<div class="kpi"><div class="icon">{_safe_text(icon)}</div><div class="val" style="color:{color};">{_safe_text(val)}</div><div class="lbl">{_safe_text(label)}</div></div>', unsafe_allow_html=True)
            if len(sessions) > 1:
                st.markdown('<div class="section-title">Session Progression</div>', unsafe_allow_html=True)
                df2 = pd.DataFrame(sessions).sort_values("id")
                fig2 = px.area(df2, x="id", y="avg_score", markers=True)
                fig2.update_traces(line_color="#8b5cf6", line_width=3, fillcolor="rgba(139,92,246,0.15)", marker=dict(size=10, color="#22d3ee"))
                fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", yaxis_range=[0, 10], height=300)
                st.plotly_chart(fig2, use_container_width=True)
        if hist:
            st.markdown('<div class="section-title">Preparation Matrix</div>', unsafe_allow_html=True)
            weak = st.session_state.manager.weak_areas if st.session_state.manager else []
            strong = st.session_state.manager.strong_areas if st.session_state.manager else []
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("**High Priority**")
                for w in (weak or ["-"]):
                    st.markdown(f'<span class="chip-gap">{_safe_text(w)}</span>', unsafe_allow_html=True)
            with m2:
                st.markdown("**Strong / Interview Ready**")
                for s in (strong or ["-"]):
                    st.markdown(f'<span class="chip-ok">{_safe_text(s)}</span>', unsafe_allow_html=True)
        if st.session_state.speech_metrics:
            st.markdown('<div class="section-title">Speech Analytics</div>', unsafe_allow_html=True)
            m = st.session_state.speech_metrics
            sa, sb, sc, sd = st.columns(4)
            speech_items = [
                (sa, "W", "WPM", m["wpm"]),
                (sb, "F", "Fillers", m["filler_total"]),
                (sc, "N", "Words", m["words"]),
                (sd, "C", "Clarity", f'{m["clarity"]}/10'),
            ]
            for col, icon, label, val in speech_items:
                with col:
                    st.markdown(f'<div class="kpi"><div class="icon">{_safe_text(icon)}</div><div class="val">{_safe_text(val)}</div><div class="lbl">{_safe_text(label)}</div></div>', unsafe_allow_html=True)

with tabs[5]:
    sessions = get_sessions()
    st.markdown(f'<div class="section-title">Session History <span class="count">{len(sessions)} SAVED</span></div>', unsafe_allow_html=True)
    if not sessions:
        st.markdown('<div class="empty"><div class="ico">EMPTY</div><div class="h">No saved sessions</div><div class="p">Finish an interview in Tab 1.</div></div>', unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search = st.text_input("Search by role", placeholder="e.g. Backend, Data...", label_visibility="collapsed")
        with c2:
            sort_by = st.selectbox("Sort", ["Newest first", "Oldest first", "Highest score", "Lowest score"], label_visibility="collapsed")
        with c3:
            confirm_wipe = st.checkbox("Enable wipe-all")
        data = list(sessions)
        if search:
            s_low = search.lower()
            data = [s for s in data if s_low in (s.get("role") or "").lower() or s_low in (s.get("interview_type") or "").lower()]
        if sort_by == "Newest first":
            data.sort(key=lambda x: x["id"], reverse=True)
        elif sort_by == "Oldest first":
            data.sort(key=lambda x: x["id"])
        elif sort_by == "Highest score":
            data.sort(key=lambda x: x.get("avg_score") or 0, reverse=True)
        elif sort_by == "Lowest score":
            data.sort(key=lambda x: x.get("avg_score") or 0)
        if confirm_wipe:
            st.warning("Wipe-all is enabled. This cannot be undone.")
            if st.button("Delete ALL Sessions", use_container_width=True):
                delete_all_sessions()
                st.success("All sessions deleted.")
                st.rerun()
        st.markdown("")
        for s in data:
            score = float(s.get("avg_score") or 0)
            s_cls = "score-hi" if score >= 7.5 else ("score-mid" if score >= 5.0 else "score-lo")
            created = (s.get("created_at") or "")[:16].replace("T", " ")
            col_info, col_score, col_view, col_del = st.columns([6, 1.2, 1, 1])
            with col_info:
                st.markdown(f'<div class="session-row"><div class="sid">#{_safe_text(s["id"])}</div><div class="meta"><div class="role">{_safe_text(s.get("role", "-"))}</div><div class="sub">{_safe_text(s.get("interview_type", "-"))} - {_safe_text(s.get("personality", "-"))} - {_safe_text(s.get("difficulty", "-"))} - {_safe_text(created)}</div></div></div>', unsafe_allow_html=True)
            with col_score:
                st.markdown(f'<div class="{s_cls}" style="text-align:center;padding:12px 0;border-radius:10px;font-weight:800;font-size:1.05rem;">{score:.1f}</div>', unsafe_allow_html=True)
            with col_view:
                if st.button("View", key=f"view_{s['id']}", use_container_width=True):
                    st.session_state["_view_session"] = s["id"]
            with col_del:
                if st.button("Delete", key=f"del_{s['id']}", use_container_width=True):
                    delete_session(s["id"])
                    st.success(f"Session #{s['id']} deleted.")
                    st.rerun()
        view_id = st.session_state.get("_view_session")
        if view_id:
            st.markdown(f'<div class="section-title">Session #{_safe_text(view_id)} Details</div>', unsafe_allow_html=True)
            turns = get_turns(view_id)
            if not turns:
                st.info("No turns recorded in this session.")
            else:
                for i, t in enumerate(turns, 1):
                    ev = t.get("evaluation", {})
                    score = float(ev.get("overall", 0))
                    with st.expander(f"Q{i} - {t['question'][:80]} - {score:.1f}/10"):
                        st.markdown(f"**Your answer:** {t['answer']}")
                        st.markdown(f"**Verdict:** {ev.get('verdict', '-')}")
                        if ev.get("ideal_answer"):
                            st.markdown(f"**Ideal:** {ev['ideal_answer']}")
                        if ev.get("suggested_best_answer"):
                            st.markdown(f"**Suggested:** {ev['suggested_best_answer']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Export this session (MD)", use_container_width=True):
                    sess = next((x for x in sessions if x["id"] == view_id), sessions[0])
                    md = build_markdown_report(sess.get("role", ""), sess.get("interview_type", ""), sess.get("personality", ""), [{"question": t["question"], "answer": t["answer"], "evaluation": t.get("evaluation", {})} for t in turns], {d: 0 for d in DIMENSIONS})
                    st.download_button("Click to download", md, file_name=f"session_{view_id}.md", mime="text/markdown", use_container_width=True)
            with c2:
                if st.button("Close details", use_container_width=True):
                    st.session_state["_view_session"] = None
                    st.rerun()
