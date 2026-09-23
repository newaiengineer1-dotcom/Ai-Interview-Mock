    # Feedback for past turns
    if st.session_state.history:
        st.markdown("### 🔍 Feedback — Most Recent Answer")
        last = st.session_state.history[-1]
        ev = last["evaluation"]

        # ---- Anti-hallucination status badge ----
        hal = ev.get("hallucination_risk", {})
        if hal.get("flagged"):
            st.error(f"⚠️ **Hallucination Risk Detected** — {hal.get('reason','')}")
        grounded = st.session_state.get("_last_groundedness")
        if grounded and not grounded.get("grounded", True):
            st.warning(
                f"🔍 **Groundedness Check Failed** — "
                f"{grounded.get('reason','')} "
                f"(Confidence: {grounded.get('confidence',0):.0%})"
            )
            if grounded.get("unsupported_claims"):
                st.caption("Unsupported claims: " +
                           ", ".join(grounded["unsupported_claims"][:5]))

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(_radar_fig(ev.get("scores", {})),
                            use_container_width=True, key="live_radar")
        with c2:
            st.markdown(
                f'<div class="card"><b style="color:#a5b4fc;">'
                f'Overall: {ev.get("overall", 0):.1f} / 10</b><br>'
                f'<i style="color:#94a3b8;">{ev.get("verdict","")}</i></div>',
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

        # ---- Suggested Best Answer (new) ----
        suggested = ev.get("suggested_best_answer", "").strip()
        if suggested:
            st.markdown("#### 💡 Suggested Best Answer (Humanized)")
            st.markdown(
                f'<div class="qa-card" style="background:rgba(250,204,21,0.08);'
                f'border-left-color:#facc15;">'
                f'<b style="color:#fde047;">What you could have said instead:</b><br>'
                f'{suggested}</div>',
                unsafe_allow_html=True)

        st.markdown("#### 🧭 Comparative Review")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="qa-card"><b>YOUR ANSWER</b><br>' +
                        last["answer"] + '</div>', unsafe_allow_html=True)
            st.markdown('<div class="qa-card ideal"><b>IDEAL MODEL ANSWER</b><br>' +
                        (ev.get("ideal_answer") or "—") + '</div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="qa-card eval"><b>AI EVALUATION</b><br>' +
                        (ev.get("verdict") or "—") + '</div>',
                        unsafe_allow_html=True)
            missed = ev.get("missed", [])
            missed_html = "<br>".join(f"• {m}" for m in missed) if missed else "—"
            st.markdown('<div class="qa-card missed"><b>WHAT WAS MISSED</b><br>' +
                        missed_html + '</div>', unsafe_allow_html=True)
