import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CategoryRetriever:
    def __init__(self, docs):
        self.docs = docs or []
        self.texts = [d["text"] for d in self.docs]
        self.cats = [d["category"] for d in self.docs]
        self.vec = None
        self.matrix = None
        if self.texts:
            self.vec = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), max_features=6000
            )
            self.matrix = self.vec.fit_transform(self.texts)

    def retrieve(self, query: str, category: str | None = None, k: int = 3):
        if self.matrix is None or not query.strip():
            return []
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self.matrix).flatten()
        order = np.argsort(-sims)
        out = []
        for i in order:
            if category and self.cats[i] != category:
                continue
            out.append({"text": self.texts[i], "category": self.cats[i],
                        "score": float(sims[i])})
            if len(out) >= k:
                break
        return out
