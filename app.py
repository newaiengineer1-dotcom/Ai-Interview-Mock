import os
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database.db import (init_db, create_session, add_turn, update_session,
                         get_sessions, get_turns)
from services.groq_service import chat, transcribe, get_client, DEFAULT_MODEL
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

# =========================================================
# Setup
# =========================================================
init_db()
st.set_page_config(page_title="InterviewAI", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --brand:#6366f1; --brand2:#8b5cf6; }
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent 60%),
              radial-gradient(1000px 500px at 90% 0%, rgba(236,72,153,0.12), transparent 60%),
              #0b1020;
}
h1,h2,h3,h4 { color:#f9fafb !important; }
p,li,label,span,div { color:#e5e7eb; }
.hero {
  background: linear-gradient(135deg,#6366f1 0%, #8b5cf6 55%, #ec4899 100%);
  border-radius: 22px; padding: 26px 32px; color:#fff;
  box-shadow: 0 24px 60px rgba(99,102,241,0.35);
  margin-bottom: 18px;
}
.hero h1 { color:#fff !important; margin:0; font-size:2.05rem; letter-spacing:-0.5px; }
.hero p { color:#eef2ff; margin-top:6px; font-size:0.98rem; }
.card {
  background: rgba(17,24,39,0.7);
  border: 1px solid rgba(99,102,241,0.22);
  border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 12px 34px rgba(0,0,0,0.35);
  backdrop-filter: blur(6px);
}
.pill { display:inline-block; padding:4px 12px; border-radius:999px;
        background:rgba(99,102,241,0.18); color:#c7d2fe;
        font-size:12px; margin:0 6px 6px 0; border:1px solid rgba(99,102,241,0.3);}
.metric { background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.06));
          border:1px solid rgba(99,102,241,0.3); border-radius:14px; padding:14px 12px; text-align:center; }
.metric .v { font-size:1.55rem; font-weight:800; color:#a5b4fc; line-height:1.1;}
.metric .l { font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px;}
.qa-card { background:rgba(99,102,241,0.07); border-left:4px solid #6366f1;
           padding:12px 16px; border-radius:10px; margin:8px 0; }
.qa-card.eval { background:rgba(16,185,129,0.06); border-left-color:#10b981; }
.qa-card.ideal { background:rgba(59,130,246,0.07); border-left-color:#3b82f6; }
.qa-card.missed { background:rgba(239,68,68,0.06); border-left-color:#ef4444; }
.chip-ok { background:rgba(16,185,129,0.15); color:#6ee7b7; padding:2px 10px; border-radius:999px; font-size:12px; margin:2px 4px 2px 0; display:inline-block;}
.chip-gap{ background:rgba(239,68,68,0.15); color:#fca5a5; padding:2px 10px; border-radius:999px; font-size:12px; margin:2px 4px 2px 0; display:inline-block;}
.stButton>button {
  background: linear-gradient(135deg,#6366f1,#8b5cf6);
  color:#fff; border:0; border-radius:10px; font-weight:600; padding:0.5rem 1.1rem;
}
.stButton>button:hover { filter:brightness(1.08); }
section[data-testid="stSidebar"] { background:#0a0f1f; border-right:1px solid rgba(99,102,241,0.15);}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Helper functions (defined BEFORE use)
# =========================================================
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
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        line=dict(color="#8b5cf6", width=2),
        fillcolor="rgba(139,92,246,0.25)",
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor="rgba(148,163,184,0.25)",
                            tickfont=dict(color="#94a3b8")),
            angularaxis=dict(tickfont=dict(color="#e5e7eb")),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        height=380,
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


# =========================================================
# Session state
# =========================================================
def _init_state():
    defaults = {
        "api_key": "",
        "manager": None,
        "session_id": None,
        "history": [],
        "current_question": None,
        "resume_text": "",
        "jd_text": "",
        "resume_sections": {},
        "jd_sections": {},
        "retriever": None,
        "role": "",
        "itype": INTERVIEW_TYPES[0],
        "personality": list(PERSONALITIES.keys())[0],
        "difficulty": "Medium",
        "analyzed": False,
        "ats_report": None,
        "speech_metrics": None,
        "coding_result": None,
        "_last_transcript": "",
        "_last_groundedness": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()

# =========================================================
# Hero
# =========================================================
st.markdown("""
<div class="hero">
  <h1>🎯 InterviewAI</h1>
  <p>Agentic, multi-modal AI interview coach — tailored to your CV & the job description. Real questions. Real feedback. Real growth.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    try:
        key_ok = bool(st.secrets.get("GROQ_API_KEY"))
    except Exception:
        key_ok = False
    if not key_ok and not os.environ.get("GROQ_API_KEY"):
        st.session_state.api_key = st.text_input(
            "Groq API Key", type="password",
            value=st.session_state.api_key,
            help="Get one free at console.groq.com"
        )
    else:
        st.success("Groq API key loaded from secrets/env.")

    st.caption(f"🧠 Model: `{DEFAULT_MODEL}`")

    st.markdown("---")
    st.markdown("### 📎 Documents")
    resume_file = st.file_uploader("Resume (PDF / DOCX / TXT)",
                                   type=["pdf", "docx", "txt"])
    jd_file = st.file_uploader("Job Description (PDF / DOCX / TXT)",
                               type=["pdf", "docx", "txt"])

    st.markdown("---")
    st.markdown("### 🎛️ Interview Setup")
    st.session_state.role = st.text_input(
        "Target Role", value=st.session_state.role or "Software Engineer")
    st.session_state.itype = st.selectbox(
        "Interview Type", INTERVIEW_TYPES,
        index=INTERVIEW_TYPES.index(st.session_state.itype))
    st.session_state.personality = st.selectbox(
        "Interviewer Personality", list(PERSONALITIES.keys()),
        index=list(PERSONALITIES.keys()).index(st.session_state.personality))
    st.session_state.difficulty = st.selectbox(
        "Starting Difficulty", DIFFICULTY_LEVELS,
        index=DIFFICULTY_LEVELS.index(st.session_state.difficulty))

    analyze_btn = st.button("🔍 Analyze Documents", use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        for k in ["manager", "session_id", "current_question", "ats_report",
                  "speech_metrics", "coding_result", "_last_groundedness"]:
            st.session_state[k] = None
        st.session_state["history"] = []
        st.session_state["analyzed"] = False
        st.session_state["_last_transcript"] = ""
        st.success("Session reset.")


# =========================================================
# Handle analyze
# =========================================================
if analyze_btn:
    with st.spinner("Parsing & indexing documents…"):
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
    st.success(f"✅ Indexed {len(docs)} chunks across resume & JD.")


# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(["🎯 Interview", "🧠 Resume & ATS", "💻 Coding",
                "📊 Dashboard", "📁 History"])


# =========================================================
# TAB 1: INTERVIEW
# =========================================================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric"><div class="v">{len(st.session_state.history)}</div>'
            f'<div class="l">Questions</div></div>',
            unsafe_allow_html=True)
    with col2:
        _scores = [h["evaluation"].get("overall", 0)
                   for h in st.session_state.history]
        _avg = sum(_scores) / len(_scores) if _scores else 0
        st.markdown(
            f'<div class="metric"><div class="v">{_avg:.1f}</div>'
            f'<div class="l">Avg Score</div></div>',
            unsafe_allow_html=True)
    with col3:
        _diff = (st.session_state.manager.difficulty
                 if st.session_state.manager else st.session_state.difficulty)
        st.markdown(
            f'<div class="metric"><div class="v">{_diff}</div>'
            f'<div class="l">Difficulty</div></div>',
            unsafe_allow_html=True)
    with col4:
        _sid = st.session_state.session_id or "—"
        st.markdown(
            f'<div class="metric"><div class="v">#{_sid}</div>'
            f'<div class="l">Session</div></div>',
            unsafe_allow_html=True)

    st.markdown("")

    # ---- Start button ----
    if st.session_state.manager is None:
        if st.button("🚀 Start Interview", use_container_width=True):
            if not st.session_state.analyzed and not st.session_state.resume_text:
                st.warning("Upload a Resume and/or JD, then click 'Analyze Documents'.")
            else:
                st.session_state.manager = InterviewManager(
                    st.session_state.difficulty)
                st.session_state.history = []
                st.session_state.session_id = create_session(
                    st.session_state.role, st.session_state.itype,
                    st.session_state.personality, st.session_state.difficulty)
                retriever = st.session_state.retriever
                r_ctx = retriever.retrieve(
                    st.session_state.role + " " + st.session_state.itype,
                    k=4) if retriever else []
                j_ctx = []
                if retriever:
                    j_ctx = [d for d in retriever.docs
                             if d["category"].startswith("jd_")][:3]
                with st.spinner("Preparing first question…"):
                    q = generate_question(
                        st.session_state.role, st.session_state.itype,
                        st.session_state.personality,
                        st.session_state.difficulty,
                        r_ctx, j_ctx, [],
                        api_key=st.session_state.api_key)
                st.session_state.current_question = q
                st.rerun()

    # ---- Question display + answer ----
    if st.session_state.manager and st.session_state.current_question:
        st.markdown(
            f'<div class="card" style="margin-bottom:12px;">'
            f'<div style="font-size:0.75rem;color:#94a3b8;'
            f'text-transform:uppercase;letter-spacing:0.1em;">'
            f'Question {len(st.session_state.history)+1} • '
            f'{st.session_state.manager.difficulty}</div>'
            f'<div style="font-size:1.15rem;color:#f9fafb;margin-top:8px;'
            f'font-weight:500;">{st.session_state.current_question}</div>'
            f'</div>',
            unsafe_allow_html=True)

        mode = st.radio("Answer mode", ["✍️ Type", "🎙️ Voice"],
                        horizontal=True, key="answer_mode")

        answer_text = ""
        speech = None

        if mode == "✍️ Type":
            answer_text = st.text_area(
                "Your answer", height=160, key="typed_answer",
                placeholder="Type your answer — be specific, concise, "
                            "and reference your own experience…")
        else:
            st.caption("Record with your mic or upload an audio file.")
            audio_data = None
            try:
                audio_data = st.audio_input("Record your answer")
            except Exception:
                pass
            uploaded_audio = st.file_uploader(
                "…or upload audio", type=["wav", "mp3", "m4a", "webm"])
            if uploaded_audio is not None:
                audio_data = uploaded_audio

            if audio_data is not None and st.button("📝 Transcribe"):
                with st.spinner("Transcribing via Whisper…"):
                    with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_data.read())
                        path = tmp.name
                    try:
                        text = transcribe(path,
                                          api_key=st.session_state.api_key)
                        st.session_state["_last_transcript"] = text
                        speech = analyze_speech(text, 30)
                        st.session_state.speech_metrics = speech
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

            answer_text = st.text_area(
                "Transcript (edit if needed)",
                value=st.session_state.get("_last_transcript", ""),
                height=140, key="voice_answer")

        colA, colB = st.columns([1, 1])
        with colA:
            submit = st.button("✅ Submit Answer", use_container_width=True)
        with colB:
            skip = st.button("⏭️ Skip", use_container_width=True)

        if submit or skip:
            if not answer_text and not skip:
                st.warning("Please provide an answer or skip.")
            else:
                ans = answer_text if answer_text else "(skipped)"
                retriever = st.session_state.retriever
                ctx_query = st.session_state.current_question + " " + ans[:200]
                r_ctx = retriever.retrieve(ctx_query, k=4) if retriever else []
                j_ctx = [d for d in (retriever.docs if retriever else [])
                         if d["category"].startswith("jd_")][:3]

                with st.spinner("Evaluating…"):
                    ev = evaluate_answer(
                        st.session_state.current_question, ans,
                        st.session_state.role, r_ctx, j_ctx,
                        api_key=st.session_state.api_key)

                with st.spinner("Running groundedness check…"):
                    grounded = check_groundedness(
                        st.session_state.current_question, ans,
                        r_ctx + j_ctx,
                        api_key=st.session_state.api_key)
                    st.session_state["_last_groundedness"] = grounded

                if not grounded.get("grounded", True):
                    ev["hallucination_risk"] = {
                        "flagged": True,
                        "reason": grounded.get("reason", ""),
                        "unsupported_claims":
                            grounded.get("unsupported_claims", []),
                    }

                st.session_state.manager.record(
                    st.session_state.current_question, ev)
                st.session_state.history.append({
                    "question": st.session_state.current_question,
                    "answer": ans,
                    "evaluation": ev,
                    "groundedness": grounded,
                })
                add_turn(st.session_state.session_id,
                         len(st.session_state.history),
                         st.session_state.current_question, ans,
                         ev, ev.get("scores", {}))

                with st.spinner("Thinking of next question…"):
                    next_q = generate_question(
                        st.session_state.role, st.session_state.itype,
                        st.session_state.personality,
                        st.session_state.manager.difficulty,
                        r_ctx, j_ctx, st.session_state.history,
                        api_key=st.session_state.api_key)
                st.session_state.current_question = next_q
                st.session_state["_last_transcript"] = ""
                st.rerun()

    # ---- Feedback for most recent answer ----
    if st.session_state.history:
        st.markdown("### 🔍 Feedback — Most Recent Answer")
        last = st.session_state.history[-1]
        ev = last["evaluation"]

        hal = ev.get("hallucination_risk", {})
        if hal.get("flagged"):
            st.error(f"⚠️ **Hallucination Risk Detected** — "
                     f"{hal.get('reason','')}")
        grounded = st.session_state.get("_last_groundedness")
        if grounded and not grounded.get("grounded", True):
            st.warning(
                f"🔍 **Groundedness Check Failed** — "
                f"{grounded.get('reason','')} "
                f"(Confidence: {grounded.get('confidence',0):.0%})")
            if grounded.get("unsupported_claims"):
                st.caption("Unsupported claims: " +
                           ", ".join(grounded["unsupported_claims"][:5]))

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(_radar_fig(ev.get("scores", {})),
                            use_container_width=True, key="live_radar")
        with c2:
            st.markdown(
                f'<div class="card">'
                f'<b style="color:#a5b4fc;">Overall: '
                f'{ev.get("overall", 0):.1f} / 10</b><br>'
                f'<i style="color:#94a3b8;">{ev.get("verdict","")}</i>'
                f'</div>',
                unsafe_allow_html=True)
            if ev.get("strengths"):
                st.markdown("**✅ Strengths**")
                st.markdown("".join(
                    f'<span class="chip-ok">{s}</span>'
                    for s in ev["strengths"]), unsafe_allow_html=True)
            if ev.get("gaps"):
                st.markdown("**⚠️ Gaps**")
                st.markdown("".join(
                    f'<span class="chip-gap">{g}</span>'
                    for g in ev["gaps"]), unsafe_allow_html=True)

        suggested = ev.get("suggested_best_answer", "").strip()
        if suggested:
            st.markdown("#### 💡 Suggested Best Answer (Humanized)")
            st.markdown(
                f'<div class="qa-card" style="background:rgba(250,204,21,0.08);'
                f'border-left-color:#facc15;">'
                f'<b style="color:#fde047;">What you could have said instead:'
                f'</b><br>{suggested}</div>',
                unsafe_allow_html=True)

        st.markdown("#### 🧭 Comparative Review")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(
                '<div class="qa-card"><b>YOUR ANSWER</b><br>' +
                last["answer"] + '</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="qa-card ideal"><b>IDEAL MODEL ANSWER</b><br>' +
                (ev.get("ideal_answer") or "—") + '</div>',
                unsafe_allow_html=True)
        with cc2:
            st.markdown(
                '<div class="qa-card eval"><b>AI EVALUATION</b><br>' +
                (ev.get("verdict") or "—") + '</div>', unsafe_allow_html=True)
            missed = ev.get("missed", [])
            missed_html = "<br>".join(f"• {m}" for m in missed) if missed else "—"
            st.markdown(
                '<div class="qa-card missed"><b>WHAT WAS MISSED</b><br>' +
                missed_html + '</div>', unsafe_allow_html=True)

        # ---- Export ----
        st.markdown("### 📥 Export Report")
        md = build_markdown_report(
            st.session_state.role, st.session_state.itype,
            st.session_state.personality, st.session_state.history,
            _summary_scores(st.session_state.history),
            st.session_state.speech_metrics)
        html = build_html_report(md)
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "⬇️ Download Markdown", md,
                file_name=f"interview_report_"
                          f"{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown", use_container_width=True)
        with ec2:
            st.download_button(
                "⬇️ Download HTML", html,
                file_name=f"interview_report_"
                          f"{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html", use_container_width=True)

        if st.button("🏁 Finish & Save Session", use_container_width=True):
            _avg2 = (sum(h["evaluation"].get("overall", 0)
                         for h in st.session_state.history)
                     / max(len(st.session_state.history), 1))
            update_session(st.session_state.session_id, _avg2,
                           summary="Session completed",
                           metrics=st.session_state.speech_metrics or {})
            st.success(f"Session #{st.session_state.session_id} saved "
                       f"(avg {_avg2:.1f}/10).")


# =========================================================
# TAB 2: RESUME & ATS
# =========================================================
with tabs[1]:
    st.markdown("### 🧠 ATS + Recruiter Analysis")
    if not st.session_state.resume_text or not st.session_state.jd_text:
        st.info("Upload both a Resume and a Job Description in the sidebar, "
                "then click **Analyze Documents**.")
    else:
        if st.button("🔎 Run ATS Gap Analysis", use_container_width=True):
            with st.spinner("Analyzing…"):
                st.session_state.ats_report = analyze_resume_jd(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                    api_key=st.session_state.api_key)
        rep = st.session_state.ats_report
        if rep:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(
                    f'<div class="metric"><div class="v">'
                    f'{rep.get("match_percent",0)}%</div>'
                    f'<div class="l">Match</div></div>',
                    unsafe_allow_html=True)
            for label, key, col in [("Skills", "technical_skills", r2),
                                    ("Projects", "projects", r3),
                                    ("Impact", "impact", r4)]:
                with col:
                    v = rep.get(key, {}).get("score", 0)
                    st.markdown(
                        f'<div class="metric"><div class="v">{v}</div>'
                        f'<div class="l">{label}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("#### 🎯 Skill Weighting (from JD)")
            sw = rep.get("skill_weights", [])
            if sw:
                df = pd.DataFrame(sw)
                fig = px.bar(df, x="weight", y="skill", orientation="h",
                             color="weight", color_continuous_scale="Purples")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e5e7eb", height=320,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            a1, a2 = st.columns(2)
            with a1:
                st.markdown("#### ✅ Strong Matches")
                st.markdown("".join(
                    f'<span class="chip-ok">{s}</span>'
                    for s in rep.get("strong_matches", [])),
                    unsafe_allow_html=True)
            with a2:
                st.markdown("#### ⚠️ Missing Gaps")
                st.markdown("".join(
                    f'<span class="chip-gap">{g}</span>'
                    for g in rep.get("missing_gaps", [])),
                    unsafe_allow_html=True)

            st.markdown("#### ✍️ Bullet Rewrites (XYZ / STAR)")
            for r in rep.get("rewrites", []):
                st.markdown(
                    f'<div class="qa-card missed">'
                    f'<b>Before:</b> {r.get("before","")}<br>'
                    f'<b>After:</b> {r.get("after","")}</div>',
                    unsafe_allow_html=True)

            st.markdown("#### 🗺️ Suggested 4-Part Interview Plan")
            for i, p in enumerate(rep.get("plan", []), 1):
                st.markdown(f"- **Part {i}:** {p}")

            st.markdown("#### 📋 Notes by Category")
            for k in ["technical_skills", "projects", "impact",
                      "ats_readiness", "role_alignment"]:
                v = rep.get(k, {})
                if v:
                    st.markdown(
                        f"**{k.replace('_',' ').title()}** — "
                        f"{v.get('score','-')}/10: {v.get('notes','')}")


# =========================================================
# TAB 3: CODING
# =========================================================
with tabs[2]:
    st.markdown("### 💻 Coding Interview Mode")
    problem = st.selectbox(
        "Choose a problem",
        [f"{p['title']}  •  {p['difficulty']}" for p in PROBLEMS])
    title = problem.split("  •  ")[0]
    prob = next(p for p in PROBLEMS if p["title"] == title)
    st.markdown(
        f'<div class="card"><b>{prob["title"]}</b> '
        f'<span class="pill">{prob["difficulty"]}</span><br>'
        f'{prob["prompt"]}<br>'
        f'<i style="color:#94a3b8;">{prob["examples"]}</i></div>',
        unsafe_allow_html=True)
    lang = st.selectbox("Language",
                        ["python", "javascript", "java", "cpp", "go"])
    code = st.text_area("Your solution", height=280,
                        placeholder="# write your solution here")
    if st.button("🧪 Evaluate Code", use_container_width=True):
        if not code.strip():
            st.warning("Write some code first.")
        else:
            with st.spinner("Analyzing complexity & correctness…"):
                res = evaluate_code(prob["title"], prob["prompt"], code, lang,
                                    api_key=st.session_state.api_key)
            st.session_state.coding_result = res
    res = st.session_state.coding_result
    if res:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric"><div class="v">'
                f'{res.get("correctness",{}).get("score",0)}</div>'
                f'<div class="l">Correctness</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric"><div class="v">'
                f'{res.get("code_quality",{}).get("score",0)}</div>'
                f'<div class="l">Code Quality</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric"><div class="v">'
                f'{res.get("overall",0)}</div>'
                f'<div class="l">Overall</div></div>',
                unsafe_allow_html=True)

        st.markdown(f"**⏱️ Time complexity:** {res.get('time_complexity','—')}")
        st.markdown(f"**💾 Space complexity:** {res.get('space_complexity','—')}")
        if res.get("edge_cases"):
            st.markdown("**🧪 Edge cases considered / needed:** " +
                        ", ".join(res["edge_cases"]))
        if res.get("follow_up"):
            st.info(f"**Follow-up:** {res['follow_up']}")
        if res.get("verdict"):
            st.success(res["verdict"])


# =========================================================
# TAB 4: DASHBOARD
# =========================================================
with tabs[3]:
    st.markdown("### 📊 Executive Analytics")
    hist = st.session_state.history
    if not hist:
        st.info("Complete at least one interview question to see analytics.")
    else:
        summary = _summary_scores(hist)
        d1, d2 = st.columns([1, 1])
        with d1:
            st.markdown("#### 🕸️ 6-Dimensional Competency Radar")
            st.plotly_chart(_radar_fig(summary),
                            use_container_width=True, key="dash_radar")
        with d2:
            st.markdown("#### 📈 Question-by-Question Trajectory")
            df = pd.DataFrame({
                "Q": [f"Q{i+1}" for i in range(len(hist))],
                "Score": [h["evaluation"].get("overall", 0) for h in hist],
            })
            fig = px.line(df, x="Q", y="Score", markers=True)
            fig.update_traces(line_color="#8b5cf6", marker=dict(size=10))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e5e7eb", yaxis_range=[0, 10],
                              height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🧭 Preparation Matrix")
        weak = (st.session_state.manager.weak_areas
                if st.session_state.manager else [])
        strong = (st.session_state.manager.strong_areas
                  if st.session_state.manager else [])
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**🔥 High Priority**")
            for w in (weak or ["—"]):
                st.markdown(f'<span class="chip-gap">{w}</span>',
                            unsafe_allow_html=True)
        with m2:
            st.markdown("**✓ Strong / Interview Ready**")
            for s in (strong or ["—"]):
                st.markdown(f'<span class="chip-ok">{s}</span>',
                            unsafe_allow_html=True)

        if st.session_state.speech_metrics:
            st.markdown("#### 🎙️ Speech Analytics (last voice answer)")
            m = st.session_state.speech_metrics
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(
                    f'<div class="metric"><div class="v">{m["wpm"]}</div>'
                    f'<div class="l">WPM</div></div>',
                    unsafe_allow_html=True)
            with s2:
                st.markdown(
                    f'<div class="metric">'
                    f'<div class="v">{m["filler_total"]}</div>'
                    f'<div class="l">Fillers</div></div>',
                    unsafe_allow_html=True)
            with s3:
                st.markdown(
                    f'<div class="metric"><div class="v">{m["words"]}</div>'
                    f'<div class="l">Words</div></div>',
                    unsafe_allow_html=True)
            with s4:
                st.markdown(
                    f'<div class="metric">'
                    f'<div class="v">{m["clarity"]}/10</div>'
                    f'<div class="l">Clarity</div></div>',
                    unsafe_allow_html=True)

        # ---- Anti-Hallucination Summary ----
        flagged = [h for h in hist
                   if h.get("evaluation", {}).get("hallucination_risk", {})
                   .get("flagged")]
        ungrounded = [h for h in hist
                      if h.get("groundedness", {}).get("grounded") is False]
        if flagged or ungrounded:
            st.markdown("#### 🛡️ Anti-Hallucination Summary")
            h1, h2 = st.columns(2)
            with h1:
                st.markdown(
                    f'<div class="metric"><div class="v">{len(flagged)}</div>'
                    f'<div class="l">Hallucination Flags</div></div>',
                    unsafe_allow_html=True)
            with h2:
                st.markdown(
                    f'<div class="metric"><div class="v">{len(ungrounded)}</div>'
                    f'<div class="l">Ungrounded Answers</div></div>',
                    unsafe_allow_html=True)


# =========================================================
# TAB 5: HISTORY
# =========================================================
with tabs[4]:
    st.markdown("### 📁 Session History (SQLite)")
    sessions = get_sessions()
    if not sessions:
        st.info("No saved sessions yet.")
    else:
        df = pd.DataFrame(sessions)[
            ["id", "created_at", "role", "interview_type",
             "personality", "difficulty", "avg_score"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(sessions) > 1:
            df2 = pd.DataFrame(sessions).sort_values("id")
            fig = px.line(df2, x="id", y="avg_score", markers=True,
                          title="Performance Progression "
                                "(avg score per session)")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e5e7eb", yaxis_range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)

        sid = st.selectbox("View session details",
                           [s["id"] for s in sessions])
        turns = get_turns(sid)
        for i, t in enumerate(turns, 1):
            ev = t.get("evaluation", {})
            with st.expander(
                    f"Q{i}: {t['question'][:90]} — "
                    f"{float(ev.get('overall',0)):.1f}/10"):
                st.markdown(f"**Your answer:** {t['answer']}")
                if ev.get("ideal_answer"):
                    st.markdown(f"**Ideal:** {ev['ideal_answer']}")
                if ev.get("suggested_best_answer"):
                    st.markdown(
                        f"**💡 Suggested:** {ev['suggested_best_answer']}")
                if ev.get("gaps"):
                    st.markdown(f"**Gaps:** {', '.join(ev['gaps'])}")
