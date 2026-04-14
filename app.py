import streamlit as st
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate
from openai import OpenAI

# ✅ Initialize OpenAI client
client = OpenAI()

st.title("RAG Chatbot - Ask My Docs")

retriever = HybridRetriever()
reranker = Reranker()

query = st.text_input("Enter your question:")

# ✅ Faithfulness scoring helper
def compute_faithfulness(query, answer, contexts):
    dataset = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts]
    })
    results = evaluate(dataset, metrics=[faithfulness])
    return results["faithfulness"][0]

# ✅ Compress multiple chunks into a coherent passage
def compress_context(docs):
    joined = "\n\n".join(docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Condense the following text into a coherent, concise summary:\n\n{joined}"
        }],
        temperature=0
    )
    return response.choices[0].message.content

# ✅ Summarize with structured output and citations
def summarize_answer(query, docs, sources):
    compressed_context = compress_context(docs)
    prompt = f"""
    You are a helpful assistant. Using ONLY the provided context, answer the question below.
    Structure your answer with:
    - Summary of findings
    - Key figures or trends
    - Source citations (filenames)

    Question: {query}
    Context: {compressed_context}

    Answer:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

if query:
    # Retrieve candidates with text + source
    candidates = retriever.retrieve(query, top_k=10)
    texts = [c[0] for c in candidates]
    sources = [c[1] for c in candidates]

    # Rerank only on text
    results = reranker.rerank(query, texts, top_k=3)
    top_docs = [doc for doc, _ in results]
    top_sources = [sources[texts.index(doc)] for doc in top_docs]

    # ✅ Summarized answer
    summary = summarize_answer(query, top_docs, top_sources)
    st.subheader("Summarized Answer")
    st.write(summary)

    # ✅ Raw Top Answers (debugging transparency)
    st.subheader("Top Answers (Raw Chunks)")
    for (doc, score) in results:
        idx = texts.index(doc)
        source = sources[idx]
        faith_score = compute_faithfulness(query, doc, [doc])

        st.write(f"Score: {score:.4f}")
        st.write(doc)
        st.write(f"Source: {source}")
        st.write(f"Faithfulness: {faith_score:.2f}")
        st.markdown("---")
