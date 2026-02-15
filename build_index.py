import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load corpus
with open("corpus.json", "r") as f:
    corpus = json.load(f)

texts = [item["text"] for item in corpus]

# SBERT for first-stage retrieval
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)

# Build FAISS index (cosine similarity via inner product)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, "legal_index.faiss")

print("✅ FAISS index built successfully")
