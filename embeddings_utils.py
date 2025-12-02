import os
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url=base)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")

def chunk_text(text, chunk_size=150, overlap=40):
    words = text.split()
    if len(words) <= chunk_size:
        return [clean_text(" ".join(words))]
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(clean_text(chunk))
        i += chunk_size - overlap
    return chunks

def embed_texts(texts):
    client = _get_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embs = [d.embedding for d in resp.data]

    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    arr = arr / norms
    return arr

class SimpleVectorStore:
    def __init__(self):
        self.vectors = None
        self.metadatas = []
        self.nn = None

    def add(self, vectors, metadatas):
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadatas.extend(metadatas)

    def build_index(self, n_neighbors=10):
        n_neighbors = min(n_neighbors, len(self.metadatas))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.nn.fit(self.vectors)

    def search(self, query_vec, top_k=6):
        if self.nn is None:
            raise ValueError("Index not built.")
        top_k = min(top_k, len(self.metadatas))
        dists, idxs = self.nn.kneighbors([query_vec], n_neighbors=top_k)

        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            sim = 1.0 - float(dist)
            results.append((self.metadatas[idx], sim))
        return results

def build_store_from_textfile(path, chunk_size=150, overlap=40, n_neighbors=12):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size, overlap)
    metadatas = [{"text": c, "source": path, "chunk_id": i} for i, c in enumerate(chunks)]

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embs = embed_texts(chunks)

    store = SimpleVectorStore()
    store.add(embs, metadatas)
    store.build_index(n_neighbors=n_neighbors)

    return store

def rerank_by_overlap(query: str, candidates: list, alpha=0.20):
    q_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    overlaps = []
    max_overlap = 0

    for md, sim in candidates:
        text = md["text"].lower()
        tok = set(re.findall(r"\w+", text))
        overlap = len(q_tokens & tok)
        overlaps.append(overlap)
        max_overlap = max(max_overlap, overlap)

    for (md, sim), overlap in zip(candidates, overlaps):
        overlap_score = (overlap / max_overlap) if max_overlap > 0 else 0.0
        combined = sim + alpha * overlap_score
        scored.append((md, sim, overlap_score, combined))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored
