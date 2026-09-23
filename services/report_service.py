import re
import html
from datetime import datetime

def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s

def build_markdown_report(role, itype, personality, turns, summary_scores, metrics=None):
    lines = [
        "# 🎯 Interview Performance Report",
        f"**Role:** {role}  ",
        f"**Interview Type:** {itype}  ",
        f"**Interviewer Style:** {personality}  ",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 📊 Overall Scorecard",
    ]
    for k, v in summary_scores.items():
        lines.append(f"- **{k}**: {v:.1f}/10")
    lines.append("")
    lines.append("## 📝 Transcript & Evaluations")
    for i, t in enumerate(turns, 1):
        ev = t.get("evaluation", {}) or {}
        lines.append(f"### Q{i}. {t.get('question','')}")
        lines.append(f"**Your answer:** {t.get('answer','')}")
        try:
            lines.append(f"**Overall:** {float(ev.get('overall', 0)):.1f}/10")
        except Exception:
            lines.append(f"**Overall:** N/A")
        if ev.get("strengths"):
            lines.append("**Strengths:** " + "; ".join(ev["strengths"]))
        if ev.get("gaps"):
            lines.append("**Gaps:** " + "; ".join(ev["gaps"]))
        if ev.get("ideal_answer"):
            lines.append(f"**Ideal answer:** {ev['ideal_answer']}")
        if ev.get("missed"):
            lines.append("**Missed:** " + "; ".join(ev["missed"]))
        lines.append("")
    if metrics:
        lines.append("## 🎙️ Speech Analytics")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)

def build_html_report(md_text: str) -> str:
    out = [
        "<html><head><meta charset='utf-8'><title>InterviewAI Report</title>",
        "<style>body{font-family:Inter,system-ui,-apple-system,sans-serif;max-width:840px;margin:40px auto;line-height:1.65;color:#1f2937;padding:0 20px}"
        "h1{color:#6366f1;border-bottom:3px solid #6366f1;padding-bottom:10px}"
        "h2{color:#4f46e5;border-bottom:1px solid #e5e7eb;padding-bottom:6px;margin-top:32px}"
        "h3{color:#111827;margin-top:24px}"
        "li{margin:4px 0}"
        "strong{color:#111827}"
        ".badge{display:inline-block;background:#eef2ff;color:#4338ca;padding:2px 10px;border-radius:999px;font-size:12px;margin-right:6px}"
        "</style></head><body>",
    ]
    for ln in md_text.split("\n"):
        s = ln.strip()
        if s.startswith("### "):
            out.append(f"<h3>{_inline(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{_inline(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{_inline(s[2:])}</h1>")
        elif s.startswith("- "):
            out.append(f"<li>{_inline(s[2:])}</li>")
        elif s:
            out.append(f"<p>{_inline(s)}</p>")
    out.append("</body></html>")
    return "\n".join(out)
