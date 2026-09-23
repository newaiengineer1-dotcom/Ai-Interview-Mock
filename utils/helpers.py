import re

FILLERS = ["um", "uh", "like", "you know", "actually", "basically",
           "literally", "sort of", "kind of", "i mean", "right", "so yeah"]

def analyze_speech(text: str, duration_sec: float):
    words = re.findall(r"\b[\w']+\b", (text or "").lower())
    wc = len(words)
    wpm = int(wc / (duration_sec / 60)) if duration_sec and duration_sec > 0 else 0
    low = " " + (text or "").lower() + " "
    fillers = {}
    for f in FILLERS:
        c = low.count(" " + f + " ")
        if c:
            fillers[f] = c
    total_fillers = sum(fillers.values())
    filler_rate = total_fillers / max(wc, 1)

    clarity = 10
    if wpm and (wpm < 100 or wpm > 180):
        clarity -= 2
    if filler_rate > 0.03:
        clarity -= 2
    if filler_rate > 0.06:
        clarity -= 2
    if wc < 30:
        clarity -= 1
    clarity = max(1, min(10, clarity))

    return {
        "words": wc,
        "wpm": wpm,
        "duration_sec": round(duration_sec or 0, 1),
        "fillers": fillers,
        "filler_total": total_fillers,
        "clarity": clarity,
    }
