import os
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database.db import (init_db, create_session, add_turn, update_session,
                         get_sessions, get_turns, delete_session,
                         delete_all_sessions, get_session_stats)
from services.groq_service import chat, transcribe, DEFAULT_MODEL
from rag.loader import read_file, split_sections, RESUME_SECTIONS, JD_SECTIONS
from rag.chunker import build_index
from rag.retriever import CategoryRetriever
from agents.interviewer import generate_question, PERSONALITIES, INTERVIEW_TYPES
from agents.evaluator import (evaluate_answer, evaluate_code,
                              check_groundedness, DIMENSIONS)
from agents.manager import InterviewManager, DIFFICULTY_LEVELS
from services.resume_service import analyze_resume_jd
from services.report_service import build_markdown_report, build_html_report
from utils.helpers import analyze_speech
from utils.coding_problems import PROBLEMS


# ============================================================
# PAGE CONFIG + THEME
# ============================================================
init_db()
st.set_page_config(page_title="InterviewAI Studio", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ---------- base ---------- */
.stApp {
  background:
    radial-gradient(1000px 500px at 12% -10%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(900px 480px at 90% -5%, rgba(139,92,246,0.10), transparent 60%),
    #0a0e17;
}
section.main > div { padding-top: 0.6rem; }
h1,h2,h3,h4 { color:#f1f5f9 !important; letter-spacing:-0.01em; }
p, li, span, label, div { color:#cbd5e1; }
hr { border-color: rgba(148,163,184,0.10); }

/* ---------- hide default streamlit chrome ---------- */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- top cockpit bar ---------- */
.cockpit-top {
  display:flex; align-items:center; justify-content:space-between;
  background: linear-gradient(90deg,#0f172a 0%, #0b1220 100%);
  border:1px solid rgba(34,211,238,0.18);
  border-radius:14px; padding:10px 16px; margin-bottom:14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.cockpit-left { display:flex; align-items:center; gap:10px; }
.cockpit-logo {
  width:34px; height:34px; border-radius:10px;
  background: linear-gradient(135deg,#22d3ee,#8b5cf6);
  display:flex; align-items:center; justify-content:center;
  font-weight:800; color:#0a0e17; font-size:15px;
}
.cockpit-title { font-weight:700; color:#f1f5f9; font-size:15px; line-height:1.1; }
.cockpit-sub   { font-size:11px; color:#64748b; letter-spacing:0.14em; text-transform:uppercase; }
.cockpit-right { display:flex; gap:8px; align-items:center; }
.chip {
  background: rgba(34,211,238,0.10); color:#67e8f9;
  border:1px solid rgba(34,211,238,0.25);
  padding:4px 10px; border-radius:999px; font-size:11px;
  letter-spacing:0.04em;
}
.chip.ok  { background: rgba(16,185,129,0.10); color:#6ee7b7; border-color: rgba(16,185,129,0.30); }
.chip.warn{ background: rgba(250,204,21,0.10); color:#fde047; border-color: rgba(250,204,21,0.30); }

/* ---------- question header ---------- */
.qhead {
  display:flex; align-items:center; justify-content:space-between;
  background: rgba(15,23,42,0.7);
  border:1px solid rgba(34,211,238,0.15);
  border-radius:12px; padding:10px 14px; margin-bottom:12px;
}
.qhead-title { font-weight:600; color:#e2e8f0; font-size:14px; }
.qhead-meta { display:flex; gap:8px; align-items:center; }
.pill-diff {
  background:linear-gradient(90deg,#ef4444,#f59e0b);
  color:#0a0e17; padding:3px 10px; border-radius:999px;
  font-size:11px; font-weight:700; letter-spacing:0.05em;
}
.pill-soft {
  background: rgba(148,163,184,0.10); color:#cbd5e1;
  padding:3px 10px; border-radius:999px; font-size:11px;
  border:1px solid rgba(148,163,184,0.18);
}

/* ---------- cards ---------- */
.panel {
  background: linear-gradient(180deg, rgba(15,23,42,0.72), rgba(11,18,32,0.72));
  border: 1px solid rgba(34,211,238,0.14);
  border-radius: 14px; padding: 16px 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
  backdrop-filter: blur(6px);
  margin-bottom: 12px;
}
.panel-title {
  font-size:11px; color:#67e8f9; letter-spacing:0.14em;
  text-transform:uppercase; margin-bottom:10px; font-weight:600;
}
.panel-sub { font-size:12px; color:#94a3b8; }

/* ---------- AI interviewer avatar block ---------- */
.ai-row { display:flex; gap:12px; align-items:center; margin-bottom:10px; }
.ai-avatar {
  width:44px; height:44px; border-radius:12px;
  background: conic-gradient(from 210deg,#22d3ee,#8b5cf6,#ec4899,#22d3ee);
  display:flex; align-items:center; justify-content:center;
  font-weight:800; color:#0a0e17; font-size:15px;
}
.ai-name  { font-weight:700; color:#f1f5f9; font-size:14px; }
.ai-role  { font-size:11px; color:#94a3b8; letter-spacing:0.06em; }
.live-dot {
  width:8px; height:8px; border-radius:50%; background:#22d3ee;
  display:inline-block; margin-right:6px;
  box-shadow: 0 0 10px #22d3ee;
  animation: pulse 1.6s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1);}
  50%     { opacity:0.5; transform:scale(1.25);}
}

/* ---------- question prompt text ---------- */
.qtext { font-size:15px; line-height:1.55; color:#e2e8f0; }

/* ---------- metric tiles ---------- */
.mrow { display:grid; grid-template-columns: repeat(4,1fr); gap:10px; margin-top:12px; }
.mtile {
  background: rgba(15,23,42,0.6);
  border:1px solid rgba(34,211,238,0.16);
  border-radius:12px; padding:12px 10px; text-align:center;
}
.mtile .v { font-size:1.35rem; font-weight:800; color:#67e8f9; line-height:1.1;}
.mtile .l { font-size:10px; color:#94a3b8; text-transform:uppercase;
            letter-spacing:0.10em; margin-top:5px; }
.mtile.ok  .v { color:#6ee7b7; }

/* ---------- mic circle (visual only) ---------- */
.mic-wrap { text-align:center; padding:14px 0 6px 0;}
.mic-circle {
  width:74px; height:74px; border-radius:50%;
  background: radial-gradient(circle at 30% 30%, #22d3ee, #0891b2 70%);
  display:inline-flex; align-items:center; justify-content:center;
  box-shadow: 0 0 0 8px rgba(34,211,238,0.08), 0 0 40px rgba(34,211,238,0.35);
  font-size:28px; color:#0a0e17;
}
.mic-label { font-size:11px; color:#94a3b8; letter-spacing:0.12em;
             text-transform:uppercase; margin-top:10px; }

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {
  background: transparent !important;
  color:#94a3b8 !important;
  font-weight:600 !important;
  font-size:13px !important;
  border-radius:10px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color:#67e8f9 !important;
  background: rgba(34,211,238,0.08) !important;
}
div[data-baseweb="tab-highlight"] { background-color:#22d3ee !important; }

/* ---------- buttons ---------- */
.stButton > button {
  background: linear-gradient(135deg,#22d3ee,#8b5cf6);
  color:#0a0e17; border:0; border-radius:10px;
  font-weight:700; padding:0.5rem 1rem;
}
.stButton > button:hover { filter:brightness(1.08); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background:#080c15;
  border-right:1px solid rgba(34,211,238,0.12);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
  color:#67e8f9 !important; font-size:12px !important;
  letter-spacing:0.14em; text-transform:uppercase;
}

/* ---------- text areas ---------- */
textarea, input {
  background: rgba(15,23,42,0.75) !important;
  border:1px solid rgba(34,211,238,0.18) !important;
  color:#e2e8f0 !important;
}

/* ---------- chips ---------- */
.chip-ok, .chip-gap, .tag {
  display:inline-block; padding:3px 10px; border-radius:999px;
  font-size:11px; margin:3px 4px 3px 0;
}
.chip-ok  { background: rgba(16,185,129,0.15); color:#6ee7b7; }
.chip-gap { background: rgba(239,68,68,0.15); color:#fca5a5; }
.tag      { background: rgba(34,211,238,0.10); color:#67e8f9;
            border:1px solid rgba(34,211,238,0.25);}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def _summary_scores(hist):
    if not hist:
        return {d: 0 for d in DIMENSIONS}
    agg = {d: 0.0 for d in DIMENSIONS}
    for h in hist:
        s = h.get("evaluation", {}).get("scores", {})
        for d in DIMENSIONS:
            agg[d] += float(s.get(d, 0))
    n = len(hist)
    return {d: agg[d] / n for d in DIMENSIONS}


def _radar_fig(scores):
    cats = DIMENSIONS
    vals = [float(scores.get(c, 0)) for c in cats]
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
        font_color="#e2e8f0", height=360,
        margin=dict(l=30, r=30, t=30, b=30))
    return fig


# ============================================================
# STATE
# ============================================================
def _init_state():
    defaults = {
        "api_key": "", "manager": None, "session_id": None,
        "history": [], "current_question": None,
        "resume_text": "", "jd_text": "",
        "resume_sections": {}, "jd_sections": {},
        "retriever": None, "role": "Backend Software Engineer",
        "itype": INTERVIEW_TYPES[0],
        "personality": "FAANG-Style",
        "difficulty": "Medium",
        "analyzed": False, "ats_report": None,
        "speech_metrics": None, "coding_result": None,
        "_last_transcript": "", "_last_groundedness": None,
        "candidate_name": "Alex Chen",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ============================================================
# TOP COCKPIT BAR
# ============================================================
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
    <span class="chip ok">● Ready</span>
    <span class="chip">Model: gpt-oss-120b</span>
    <span class="chip">Difficulty: {diff}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🔑 Groq API")
    try:
        key_ok = bool(st.secrets.get("GROQ_API_KEY"))
    except Exception:
        key_ok = False
    if not key_ok and not os.environ.get("GROQ_API_KEY"):
        st.session_state.api_key = st.text_input(
            "API Key", type="password",
            value=st.session_state.api_key,
            placeholder="gsk_...", label_visibility="collapsed")
        st.caption("Get a free key at console.groq.com")
    else:
        st.success("API key loaded ✓")
    st.caption(f"🧠 `{DEFAULT_MODEL}` · Whisper-v3")

    st.markdown("---")
    st.markdown("### 👤 Candidate Setup")
    st.session_state.candidate_name = st.text_input(
        "Name", value=st.session_state.candidate_name)
    st.session_state.role = st.text_input(
        "Target Role", value=st.session_state.role)
    st.session_state.itype = st.selectbox(
        "Interview Type", INTERVIEW_TYPES,
        index=INTERVIEW_TYPES.index(st.session_state.itype))
    st.session_state.personality = st.selectbox(
        "Interviewer Persona", list(PERSONALITIES.keys()),
        index=list(PERSONALITIES.keys()).index(st.session_state.personality))
    st.session_state.difficulty = st.selectbox(
        "Starting Difficulty", DIFFICULTY_LEVELS,
        index=DIFFICULTY_LEVELS.index(st.session_state.difficulty))

    st.markdown("---")
    st.markdown("### 📎 RAG & ATS")
    resume_file = st.file_uploader("Resume (PDF/DOCX/TXT)",
                                   type=["pdf", "docx", "txt"])
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)",
                               type=["pdf", "docx", "txt"])
    analyze_btn = st.button("🔍 Analyze Documents", use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        for k in ["manager", "session_id", "current_question",
                  "ats_report", "speech_metrics", "coding_result",
                  "_last_groundedness"]:
            st.session_state[k] = None
        st.session_state["history"] = []
        st.session_state["analyzed"] = False
        st.session_state["_last_transcript"] = ""
        st.rerun()


# ---- analyze handler ----
if analyze_btn:
    with st.spinner("Parsing & indexing…"):
        if resume_file:
            st.session_state.resume_text = read_file(resume_file)
            st.session_state.resume_sections = split_sections(
                st.session_state.resume_text, RESUME_SECTIONS)
        if jd_file:
            st.session_state.jd_text = read_file(jd_file)
            st.session_state.jd_sections = split_sections(
                st.session_state.jd_text, JD_SECTIONS)
        combined = {}
        for k, v in st.session_state.resume_sections.items():
            if v:
                combined[f"resume_{k}"] = v
        for k, v in st.session_state.jd_sections.items():
            if v:
                combined[f"jd_{k}"] = v
        docs = build_index(combined)
        st.session_state.retriever = CategoryRetriever(docs)
        st.session_state.analyzed = True
    st.success(f"✅ Indexed {len(docs)} chunks.")


# ============================================================
# TABS
# ============================================================
tabs = st.tabs(["🎙️ 1. Live Interview", "📊 2. Evaluation",
                "🧠 3. Resume & ATS", "💻 4. Coding IDE",
                "📈 5. Dashboard", "📁 6. History"])


# ============================================================
# TAB 1 — LIVE INTERVIEW
# ============================================================
with tabs[0]:
    # --- question header ---
    st.markdown(f"""
    <div class="qhead">
      <div class="qhead-title">Question {q_now} of {total_q} · {st.session_state.itype}</div>
      <div class="qhead-meta">
        <span class="pill-diff">{diff}</span>
        <span class="pill-soft">{progress_pct}% complete</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- start button ---
    if st.session_state.manager is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("▶ Start Adaptive Session", use_container_width=True):
                if not st.session_state.analyzed and not st.session_state.resume_text:
                    st.warning("Upload Resume and/or JD first, then Analyze.")
                else:
                    st.session_state.manager = InterviewManager(
                        st.session_state.difficulty)
                    st.session_state.history = []
                    st.session_state.session_id = create_session(
                        st.session_state.role, st.session_state.itype,
                        st.session_state.personality, st.session_state.difficulty)
                    rt = st.session_state.retriever
                    r_ctx = rt.retrieve(st.session_state.role, k=4) if rt else []
                    j_ctx = ([d for d in rt.docs if d["category"].startswith("jd_")][:3]
                             if rt else [])
                    with st.spinner("Preparing first question…"):
                        q = generate_question(
                            st.session_state.role, st.session_state.itype,
                            st.session_state.personality, st.session_state.difficulty,
                            r_ctx, j_ctx, [], api_key=st.session_state.api_key)
                    st.session_state.current_question = q
                    st.rerun()

    # --- main two-column studio layout ---
    if st.session_state.manager and st.session_state.current_question:
        left, right = st.columns([1.05, 1], gap="large")

        # ---------------- LEFT: AI question panel ----------------
        with left:
            st.markdown(f"""
            <div class="panel">
              <div class="ai-row">
                <div class="ai-avatar">AI</div>
                <div>
                  <div class="ai-name">
                    <span class="live-dot"></span>Aria-6X
                  </div>
                  <div class="ai-role">{st.session_state.personality} · {diff} tier</div>
                </div>
              </div>
              <div class="panel-title">Question</div>
              <div class="qtext">{st.session_state.current_question}</div>
            </div>
            """, unsafe_allow_html=True)

            # Active evaluation focus chips
            weak = (st.session_state.manager.weak_areas[-3:]
                    if st.session_state.manager else [])
            if weak:
                chips = "".join(
                    f'<span class="tag">{w}</span>' for w in weak)
                st.markdown(f"""
                <div class="panel">
                  <div class="panel-title">Active Evaluation Focus</div>
                  {chips}
                </div>
                """, unsafe_allow_html=True)

            # Senior tip
            st.markdown(f"""
            <div class="panel">
              <div class="panel-title">💡 Interviewer Tip</div>
              <div class="panel-sub">
                Answer in 30–60 seconds. Reference a specific project,
                the tech stack, and one measurable outcome (latency, scale,
                or impact). Avoid buzzwords — be concrete.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- RIGHT: answer + live metrics ----------------
        with right:
            mode = st.radio("Answer Mode", ["🎙️ Voice", "⌨️ Text"],
                            horizontal=True, key="answer_mode",
                            label_visibility="collapsed")

            answer_text = ""

            if mode == "🎙️ Voice":
                st.markdown("""
                <div class="panel">
                  <div class="panel-title">Live Neural Audio Buffer</div>
                  <div class="mic-wrap">
                    <div class="mic-circle">🎤</div>
                    <div class="mic-label">Click below to record</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                audio_data = None
                try:
                    audio_data = st.audio_input("Record", label_visibility="collapsed")
                except Exception:
                    pass
                up = st.file_uploader("…or upload audio",
                                      type=["wav", "mp3", "m4a", "webm"],
                                      label_visibility="collapsed")
                if up is not None:
                    audio_data = up

                if audio_data is not None and st.button("📝 Transcribe via Whisper"):
                    with st.spinner("Transcribing…"):
                        with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_data.read())
                            path = tmp.name
                        try:
                            text = transcribe(path, api_key=st.session_state.api_key)
                            st.session_state["_last_transcript"] = text
                            st.session_state.speech_metrics = analyze_speech(text, 30)
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")

                answer_text = st.text_area(
                    "Transcript", value=st.session_state.get("_last_transcript", ""),
                    height=130, key="voice_answer",
                    placeholder="Your transcript will appear here — edit if needed…")

            else:
                answer_text = st.text_area(
                    "Your Answer", height=260, key="typed_answer",
                    placeholder="Type your answer — be specific, concise, "
                                "reference your real experience…")

            # live metrics row
            m = st.session_state.speech_metrics or {
                "wpm": 0, "filler_total": 0, "duration_sec": 0, "clarity": 0,
                "words": 0,
            }
            st.markdown(f"""
            <div class="mrow">
              <div class="mtile"><div class="v">{m.get("wpm",0)}</div>
                <div class="l">Speaking Pace</div></div>
              <div class="mtile"><div class="v">{m.get("filler_total",0)}</div>
                <div class="l">Filler Words</div></div>
              <div class="mtile"><div class="v">{m.get("duration_sec",0)}s</div>
                <div class="l">Duration</div></div>
              <div class="mtile ok"><div class="v">{m.get("clarity",0)}/10</div>
                <div class="l">Clarity Score</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            b1, b2, b3 = st.columns([1, 1, 1.4])
            with b1:
                submit = st.button("✅ Submit", use_container_width=True)
            with b2:
                skip = st.button("⏭ Skip", use_container_width=True)
            with b3:
                finish = st.button("🏁 Finish & Save", use_container_width=True)

            # ---- SUBMIT ----
            if submit or skip:
                if not answer_text and not skip:
                    st.warning("Provide an answer or skip.")
                else:
                    ans = answer_text if answer_text else "(skipped)"
                    rt = st.session_state.retriever
                    q = st.session_state.current_question
                    r_ctx = rt.retrieve(q + " " + ans[:200], k=4) if rt else []
                    j_ctx = ([d for d in rt.docs if d["category"].startswith("jd_")][:3]
                             if rt else [])

                    with st.spinner("Evaluating…"):
                        ev = evaluate_answer(q, ans, st.session_state.role,
                                             r_ctx, j_ctx,
                                             api_key=st.session_state.api_key)
                    with st.spinner("Groundedness check…"):
                        grounded = check_groundedness(q, ans, r_ctx + j_ctx,
                                                     api_key=st.session_state.api_key)
                        st.session_state["_last_groundedness"] = grounded
                    if not grounded.get("grounded", True):
                        ev["hallucination_risk"] = {
                            "flagged": True,
                            "reason": grounded.get("reason", ""),
                            "unsupported_claims":
                                grounded.get("unsupported_claims", []),
                        }

                    st.session_state.manager.record(q, ev)
                    st.session_state.history.append({
                        "question": q, "answer": ans,
                        "evaluation": ev, "groundedness": grounded})
                    add_turn(st.session_state.session_id,
                             len(st.session_state.history),
                             q, ans, ev, ev.get("scores", {}))

                    with st.spinner("Next question…"):
                        next_q = generate_question(
                            st.session_state.role, st.session_state.itype,
                            st.session_state.personality,
                            st.session_state.manager.difficulty,
                            r_ctx, j_ctx, st.session_state.history,
                            api_key=st.session_state.api_key)
                    st.session_state.current_question = next_q
                    st.session_state["_last_transcript"] = ""
                    st.rerun()

            # ---- FINISH ----
            if finish and st.session_state.history:
                avg = (sum(h["evaluation"].get("overall", 0)
                           for h in st.session_state.history)
                       / max(len(st.session_state.history), 1))
                update_session(st.session_state.session_id, avg,
                               summary="Completed",
                               metrics=st.session_state.speech_metrics or {})
                st.success(f"Session #{st.session_state.session_id} saved · avg {avg:.1f}/10")

        # ---------------- Feedback under the fold ----------------
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 🔍 Feedback — Latest Answer")
            last = st.session_state.history[-1]
            ev = last["evaluation"]

            hal = ev.get("hallucination_risk", {})
            if hal.get("flagged"):
                st.error(f"⚠️ Hallucination risk — {hal.get('reason','')}")
            g = st.session_state.get("_last_groundedness")
            if g and not g.get("grounded", True):
                st.warning(f"🔍 Ungrounded — {g.get('reason','')} "
                           f"({g.get('confidence',0):.0%})")

            f1, f2 = st.columns([1, 2])
            with f1:
                st.plotly_chart(_radar_fig(ev.get("scores", {})),
                                use_container_width=True, key="live_radar")
            with f2:
                st.markdown(f"""
                <div class="panel">
                  <div class="panel-title">Overall Score</div>
                  <div style="font-size:2rem;font-weight:800;color:#67e8f9;">
                    {ev.get("overall",0):.1f}<span style="font-size:1rem;color:#94a3b8;">/10</span>
                  </div>
                  <div class="panel-sub" style="margin-top:6px;">{ev.get("verdict","")}</div>
                </div>
                """, unsafe_allow_html=True)

                if ev.get("strengths"):
                    st.markdown("**✅ Strengths**")
                    st.markdown("".join(
                        f'<span class="chip-ok">{s}</span>' for s in ev["strengths"]),
                        unsafe_allow_html=True)
                if ev.get("gaps"):
                    st.markdown("**⚠️ Gaps**")
                    st.markdown("".join(
                        f'<span class="chip-gap">{s}</span>' for s in ev["gaps"]),
                        unsafe_allow_html=True)

            suggested = ev.get("suggested_best_answer", "").strip()
            if suggested:
                st.markdown(f"""
                <div class="panel" style="border-color:rgba(250,204,21,0.35);">
                  <div class="panel-title" style="color:#fde047;">
                    💡 Suggested Best Answer (Humanized)
                  </div>
                  <div class="qtext">{suggested}</div>
                </div>
                """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="panel"><div class="panel-title">Your Answer</div>
                <div class="panel-sub">{last["answer"]}</div></div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="panel"><div class="panel-title">Ideal Model Answer</div>
                <div class="panel-sub">{ev.get("ideal_answer") or "—"}</div></div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="panel"><div class="panel-title">What Was Missed</div>
                <div class="panel-sub">{"<br>".join("• "+m for m in ev.get("missed",[])) or "—"}</div></div>
                """, unsafe_allow_html=True)

            # export
            md = build_markdown_report(
                st.session_state.role, st.session_state.itype,
                st.session_state.personality, st.session_state.history,
                _summary_scores(st.session_state.history),
                st.session_state.speech_metrics)
            html = build_html_report(md)
            e1, e2 = st.columns(2)
            with e1:
                st.download_button("⬇ Markdown Report", md,
                                   file_name=f"report_{datetime.utcnow():%Y%m%d_%H%M}.md",
                                   mime="text/markdown",
                                   use_container_width=True)
            with e2:
                st.download_button("⬇ HTML Report", html,
                                   file_name=f"report_{datetime.utcnow():%Y%m%d_%H%M}.html",
                                   mime="text/html",
                                   use_container_width=True)


# ============================================================
# TAB 2 — EVALUATION
# ============================================================
with tabs[1]:
    st.markdown("### 📊 6-Dimensional Evaluation")
    hist = st.session_state.history
    if not hist:
        st.info("Complete at least one question first.")
    else:
        summary = _summary_scores(hist)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(_radar_fig(summary),
                            use_container_width=True, key="eval_radar")
        with c2:
            df = pd.DataFrame({
                "Dimension": list(summary.keys()),
                "Score": list(summary.values())})
            fig = px.bar(df, x="Score", y="Dimension", orientation="h",
                         color="Score", color_continuous_scale="Teal")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", height=360,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        for i, h in enumerate(hist, 1):
            ev = h["evaluation"]
            with st.expander(f"Q{i} · {h['question'][:80]} — {ev.get('overall',0):.1f}/10"):
                st.markdown(f"**Your answer:** {h['answer']}")
                st.markdown(f"**Verdict:** {ev.get('verdict','')}")
                if ev.get("ideal_answer"):
                    st.markdown(f"**Ideal:** {ev['ideal_answer']}")
                if ev.get("suggested_best_answer"):
                    st.markdown(f"**💡 Suggested:** {ev['suggested_best_answer']}")


# ============================================================
# TAB 3 — RESUME & ATS
# ============================================================
with tabs[2]:
    st.markdown("### 🧠 Resume & ATS Gap Analyzer")
    if not st.session_state.resume_text or not st.session_state.jd_text:
        st.info("Upload Resume + JD in the sidebar, then click **Analyze Documents**.")
    else:
        if st.button("🔎 Run ATS Analysis", use_container_width=True):
            with st.spinner("Analyzing…"):
                st.session_state.ats_report = analyze_resume_jd(
                    st.session_state.resume_text, st.session_state.jd_text,
                    api_key=st.session_state.api_key)
        rep = st.session_state.ats_report
        if rep:
            k1, k2, k3, k4 = st.columns(4)
            for col, label, val in [
                (k1, "Match %", f'{rep.get("match_percent",0)}%'),
                (k2, "Skills", rep.get("technical_skills",{}).get("score",0)),
                (k3, "Projects", rep.get("projects",{}).get("score",0)),
                (k4, "Impact", rep.get("impact",{}).get("score",0))]:
                with col:
                    st.markdown(f"""
                    <div class="mtile"><div class="v">{val}</div>
                    <div class="l">{label}</div></div>
                    """, unsafe_allow_html=True)

            sw = rep.get("skill_weights", [])
            if sw:
                st.markdown("#### 🎯 Skill Weighting (from JD)")
                df = pd.DataFrame(sw)
                fig = px.bar(df, x="weight", y="skill", orientation="h",
                             color="weight", color_continuous_scale="Teal")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e2e8f0", height=320,
                                  coloraxis_showscale=False,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Strong Matches**")
                st.markdown("".join(f'<span class="chip-ok">{s}</span>'
                                    for s in rep.get("strong_matches", [])),
                            unsafe_allow_html=True)
            with c2:
                st.markdown("**⚠️ Missing Gaps**")
                st.markdown("".join(f'<span class="chip-gap">{s}</span>'
                                    for s in rep.get("missing_gaps", [])),
                            unsafe_allow_html=True)

            for r in rep.get("rewrites", []):
                st.markdown(f"""
                <div class="panel">
                  <div class="panel-title">Rewrite</div>
                  <div class="panel-sub"><b>Before:</b> {r.get("before","")}</div>
                  <div class="panel-sub" style="margin-top:6px;">
                    <b style="color:#6ee7b7;">After:</b> {r.get("after","")}</div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# TAB 4 — CODING IDE
# ============================================================
with tabs[3]:
    st.markdown("### 💻 Live Coding IDE")
    problem = st.selectbox("Problem",
                           [f"{p['title']}  •  {p['difficulty']}" for p in PROBLEMS])
    title = problem.split("  •  ")[0]
    prob = next(p for p in PROBLEMS if p["title"] == title)

    st.markdown(f"""
    <div class="panel">
      <div class="panel-title">Problem · {prob["difficulty"]}</div>
      <div class="qtext">{prob["prompt"]}</div>
      <div class="panel-sub" style="margin-top:8px;">
        <b>Examples:</b> {prob["examples"]}</div>
    </div>
    """, unsafe_allow_html=True)

    lang = st.selectbox("Language", ["python", "javascript", "java", "cpp", "go"])
    code = st.text_area("Solution", height=260,
                        placeholder="# write your solution here")
    if st.button("🧪 Evaluate Code", use_container_width=True):
        if not code.strip():
            st.warning("Write some code first.")
        else:
            with st.spinner("Analyzing…"):
                st.session_state.coding_result = evaluate_code(
                    prob["title"], prob["prompt"], code, lang,
                    api_key=st.session_state.api_key)
    res = st.session_state.coding_result
    if res:
        c1, c2, c3 = st.columns(3)
        for col, label, val in [
            (c1, "Correctness", res.get("correctness",{}).get("score",0)),
            (c2, "Code Quality", res.get("code_quality",{}).get("score",0)),
            (c3, "Overall", res.get("overall",0))]:
            with col:
                st.markdown(f"""
                <div class="mtile"><div class="v">{val}</div>
                <div class="l">{label}</div></div>
                """, unsafe_allow_html=True)

        st.markdown(f"**⏱ Time:** {res.get('time_complexity','—')}")
        st.markdown(f"**💾 Space:** {res.get('space_complexity','—')}")
        if res.get("edge_cases"):
            st.markdown("**🧪 Edge cases:** " + ", ".join(res["edge_cases"]))
        if res.get("follow_up"):
            st.info(f"**Follow-up:** {res['follow_up']}")
        if res.get("verdict"):
            st.success(res["verdict"])


# ============================================================
# TAB 5 — DASHBOARD
# ============================================================
with tabs[4]:
    st.markdown("### 📈 Executive Dashboard")
    hist = st.session_state.history
    if not hist:
        st.info("Complete at least one question first.")
    else:
        summary = _summary_scores(hist)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🕸 Competency Radar")
            st.plotly_chart(_radar_fig(summary),
                            use_container_width=True, key="dash_radar")
        with c2:
            st.markdown("#### 📊 Score Trajectory")
            df = pd.DataFrame({
                "Q": [f"Q{i+1}" for i in range(len(hist))],
                "Score": [h["evaluation"].get("overall", 0) for h in hist]})
            fig = px.line(df, x="Q", y="Score", markers=True)
            fig.update_traces(line_color="#22d3ee", marker=dict(size=10))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", yaxis_range=[0,10],
                              height=360)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🧭 Preparation Matrix")
        weak = st.session_state.manager.weak_areas if st.session_state.manager else []
        strong = st.session_state.manager.strong_areas if st.session_state.manager else []
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**🔥 High Priority**")
            for w in (weak or ["—"]):
                st.markdown(f'<span class="chip-gap">{w}</span>',
                            unsafe_allow_html=True)
        with m2:
            st.markdown("**✓ Strong**")
            for s in (strong or ["—"]):
                st.markdown(f'<span class="chip-ok">{s}</span>',
                            unsafe_allow_html=True)

        if st.session_state.speech_metrics:
            st.markdown("#### 🎙 Speech Analytics")
            m = st.session_state.speech_metrics
            for col, label, val in [
                (st.columns(4)[0], "WPM", m["wpm"]),
                (st.columns(4)[1], "Fillers", m["filler_total"]),
                (st.columns(4)[2], "Words", m["words"]),
                (st.columns(4)[3], "Clarity", f'{m["clarity"]}/10')]:
                with col:
                    st.markdown(f"""
                    <div class="mtile"><div class="v">{val}</div>
                    <div class="l">{label}</div></div>
                    """, unsafe_allow_html=True)


# ============================================================
# TAB 6 — HISTORY
# ============================================================
with tabs[5]:
    st.markdown("### 📁 Session History")
    sessions = get_sessions()
    if not sessions:
        st.info("No saved sessions yet.")
    else:
        df = pd.DataFrame(sessions)[[
            "id", "created_at", "role", "interview_type",
            "personality", "difficulty", "avg_score"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(sessions) > 1:
            df2 = pd.DataFrame(sessions).sort_values("id")
            fig = px.line(df2, x="id", y="avg_score", markers=True,
                          title="Performance Progression")
            fig.update_traces(line_color="#8b5cf6")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0", yaxis_range=[0,10])
            st.plotly_chart(fig, use_container_width=True)

        sid = st.selectbox("View session", [s["id"] for s in sessions])
        for i, t in enumerate(get_turns(sid), 1):
            ev = t.get("evaluation", {})
            with st.expander(f"Q{i}: {t['question'][:80]} — {float(ev.get('overall',0)):.1f}/10"):
                st.markdown(f"**Answer:** {t['answer']}")
                if ev.get("ideal_answer"):
                    st.markdown(f"**Ideal:** {ev['ideal_answer']}")
                if ev.get("suggested_best_answer"):
                    st.markdown(f"**💡 Suggested:** {ev['suggested_best_answer']}")
