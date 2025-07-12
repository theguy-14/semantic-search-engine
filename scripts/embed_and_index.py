import json
import os
import numpy as np
import torch
import faiss
import re  # Add this import
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

# SECTION WEIGHTING FUNCTION - ADD THIS
def get_section_weight(text):
    """Assign higher weight to important sections"""
    text_lower = text.lower()
    
    # Higher weight for key sections
    if re.search(r'\b(introduction|conclusion|abstract|summary)\b', text_lower):
        return 1.5
    
    # Medium weight for methodology/results
    if re.search(r'\b(method|experiment|result|analysis|discussion)\b', text_lower):
        return 1.2
    
    # Default weight
    return 1.0

# Load chunks
with open(CHUNK_PATH, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

embeddings = []
metadata_store = []

print("\n Generating embeddings using Longformer...\n")

# Process in batches
batch_size = 4
for i in tqdm(range(0, len(chunks_data), batch_size), desc="Embedding"):
    batch = chunks_data[i:i+batch_size]
    texts = [item["text"] for item in batch]
    
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=4096
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embeddings
    cls_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
    
    # Apply section weighting - MODIFIED SECTION
    for emb, item in zip(cls_embeddings, batch):
        weight = get_section_weight(item["text"])
        weighted_emb = emb * weight
        embeddings.append(weighted_emb)
        
        metadata_store.append({
            "chunk_id": item["chunk_id"],
            "document": item["document"],
            "text": item["text"],
            "pdf_path": f"pdfs/{item['document']}.pdf"
        })

# Save FAISS index
dimension = 768
index = faiss.IndexFlatIP(dimension)
embeddings_array = np.array(embeddings).astype("float32")

# Normalize embeddings
faiss.normalize_L2(embeddings_array)
index.add(embeddings_array)

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)

# Save metadata
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_store, f, indent=2, ensure_ascii=False)

print(f"\n Saved {len(embeddings)} embeddings to FAISS index!")