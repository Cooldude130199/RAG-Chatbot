
import pdfplumber, os, pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

def load_pdf(path):
    with pdfplumber.open(path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text

def ingest_docs(data_dir="data/raw/"):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files in {data_dir}")

    texts = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for result in tqdm(ex.map(load_pdf, files), total=len(files), desc="Loading PDFs"):
            texts.append(result)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    # ✅ Attach filename metadata to each chunk
    for text, file in zip(texts, files):
        docs = splitter.create_documents([text])
        for d in docs:
            d.metadata["source"] = os.path.basename(file)
        chunks.extend(docs)

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Saved {len(chunks)} chunks with source metadata")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embeddings = embedder.encode([d.page_content for d in chunks], batch_size=32, show_progress_bar=True)

    np.save("data/embeddings.npy", embeddings)
    index = faiss.IndexFlatL2(embedder.get_embedding_dimension())
    index.add(np.array(embeddings))
    faiss.write_index(index, "data/faiss.index")

    print("✅ Cached embeddings and FAISS index")

if __name__ == "__main__":
    ingest_docs()
