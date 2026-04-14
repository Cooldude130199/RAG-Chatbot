from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        print("[Reranker] CrossEncoder loaded")

    def rerank(self, query, candidates, top_k=3):
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
