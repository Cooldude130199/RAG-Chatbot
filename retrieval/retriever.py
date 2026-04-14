import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self):
        # Load cached chunks
        with open("data/chunks.pkl", "rb") as f:
            self.docs = pickle.load(f)
        self.texts = [d.page_content for d in self.docs]
        self.sources = [d.metadata.get("source", "unknown") for d in self.docs]

        print(f"[Retriever] Loaded {len(self.texts)} chunks")

        # ✅ BM25 index
        self.bm25 = BM25Okapi([t.split() for t in self.texts])
        print("[Retriever] BM25 index ready")

        # ✅ Embeddings + FAISS index
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.embeddings = np.load("data/embeddings.npy")
        self.index = faiss.read_index("data/faiss.index")
        print("[Retriever] FAISS index loaded")

    def retrieve(self, query, top_k=5):
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())

        # FAISS vector search
        vec = self.embedder.encode([query])
        _, idxs = self.index.search(np.array(vec), top_k)

        # Hybrid union of FAISS + BM25
        hybrid = set(idxs[0]) | set(np.argsort(bm25_scores)[-top_k:])
        return [(self.texts[i], self.sources[i]) for i in hybrid]
