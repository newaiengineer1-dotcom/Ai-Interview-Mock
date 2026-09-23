DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard", "Expert"]

class InterviewManager:
    def __init__(self, start="Medium"):
        self.difficulty = start if start in DIFFICULTY_LEVELS else "Medium"
        self.recent_scores = []
        self.weak_areas = []
        self.strong_areas = []
        self.skills_tested = []
        self.asked = []

    def record(self, question, evaluation):
        self.asked.append(question)
        try:
            score = float(evaluation.get("overall", 5))
        except Exception:
            score = 5.0
        self.recent_scores.append(score)
        self.recent_scores = self.recent_scores[-5:]

        for g in evaluation.get("gaps", [])[:3]:
            if g and g not in self.weak_areas:
                self.weak_areas.append(g)
        for s in evaluation.get("strengths", [])[:3]:
            if s and s not in self.strong_areas:
                self.strong_areas.append(s)

        avg = sum(self.recent_scores) / len(self.recent_scores)
        idx = DIFFICULTY_LEVELS.index(self.difficulty)
        if avg >= 8.0 and idx < len(DIFFICULTY_LEVELS) - 1:
            self.difficulty = DIFFICULTY_LEVELS[idx + 1]
        elif avg < 6.0 and idx > 0:
            self.difficulty = DIFFICULTY_LEVELS[idx - 1]
        return avg

    def state(self):
        avg = round(sum(self.recent_scores) / len(self.recent_scores), 2) if self.recent_scores else 0.0
        return {
            "difficulty": self.difficulty,
            "avg_recent": avg,
            "weak_areas": self.weak_areas[-6:],
            "strong_areas": self.strong_areas[-6:],
        }
