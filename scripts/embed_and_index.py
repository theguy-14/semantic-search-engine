# scripts/embed_and_index.py

import json
import os
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Paths
CHUNK_PATH = "data/all_chunks.json"
INDEX_PATH = "faiss_index/chunks.index"
METADATA_PATH = "data/metadata.json"

# Load Longformer
MODEL_NAME = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load chunks
with open(CHUNK_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = []
metadata_store = []

print("\n Generating embeddings using Longformer...\n")

for chunk in tqdm(chunks):
    text = chunk["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    # Mean pooling over token embeddings
    hidden_states = output.last_hidden_state.squeeze(0)  # [seq_len, 768]
    embedding = hidden_states.mean(dim=0).cpu().numpy()  # [768]

    embeddings.append(embedding)

    metadata_store.append({
        "chunk_id": chunk["chunk_id"],
        "document": chunk["document"],
        "text": chunk["text"],
        "pdf_path": f"pdfs/{chunk['document']}.pdf"
    })

# Save FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)

# Save metadata
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_store, f, indent=2, ensure_ascii=False)

print("\n Embeddings generated and FAISS index saved!\n")
