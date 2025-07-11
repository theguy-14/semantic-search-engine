# scripts/retrieve_db.py

import json
import os
import torch
import faiss
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

# -------------------------------
# CONFIG
# -------------------------------
load_dotenv()

INDEX_PATH = "faiss_index/chunks.index"
METADATA_PATH = "data/metadata.json"
EMBED_MODEL_NAME = "allenai/longformer-base-4096"
GEN_LLM_MODEL = "EleutherAI/gpt-neo-125M"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# INITIALIZATION (runs once)
# -------------------------------

print("Loading embedding model (Longformer)...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
embed_model.eval()

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loading local LLM for answer generation...")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_LLM_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
gen_model.to(device)
generator = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer, device=0 if torch.cuda.is_available() else -1)

# -------------------------------
# FUNCTIONS
# -------------------------------

def embed_query(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = embed_model(**inputs)
    hidden_states = output.last_hidden_state.squeeze(0)
    embedding = hidden_states.mean(dim=0).cpu().numpy()
    return embedding.astype("float32").reshape(1, -1)

def retrieve_chunks(query_text, top_k=10):
    query_embedding = embed_query(query_text)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = metadata[idx]
        retrieved.append({
            "distance": float(dist),
            "document_pdf": chunk["pdf_path"],
            "chunk_id": chunk["chunk_id"],
            "snippet": chunk["text"]
        })
    return retrieved

def build_prompt(query, retrieved_chunks):
    prompt = "Use the following excerpts from PDF documents to answer the question as accurately as possible.\n\n"
    prompt += "Excerpts:\n"
    for item in retrieved_chunks:
        prompt += f"- From {item['document_pdf']} (chunk {item['chunk_id']}): \"{item['snippet']}\"\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    return prompt

def generate_answer(prompt, max_tokens=300):
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    if os.getenv("HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("HTTP_REFERER")
    if os.getenv("X_TITLE"):
        headers["X-Title"] = os.getenv("X_TITLE")

    body = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.RequestException as e:
        print(f"❌ DeepSeek API Error: {e}")
        return "Error: Failed to generate response from DeepSeek."
