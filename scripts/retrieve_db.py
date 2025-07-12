# retrieval_engine.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from google import genai
import json
import faiss

# -------------------------------
# CONFIG

INDEX_PATH = "faiss_index/chunks.index"
METADATA_PATH = "data/metadata.json"
EMBED_MODEL_NAME = "allenai/longformer-base-4096"
# GEN_LLM_MODEL = "EleutherAI/gpt-neo-125M"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()
client = genai.Client()

# -------------------------------
# INITIALIZATION 

print("Loading embedding model (Longformer)...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
embed_model.eval()



# print("Loading local LLM for answer generation...")
# gen_tokenizer = AutoTokenizer.from_pretrained(GEN_LLM_MODEL)
# gen_model = AutoModelForCausalLM.from_pretrained(GEN_LLM_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
# gen_model.to(device)
# generator = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer, device=0 if torch.cuda.is_available() else -1)

# -------------------------------
# FUNCTIONS TO INTERACT WITH THE RETRIEVAL ENGINE

def embed_query(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = embed_model(**inputs)
    hidden_states = output.last_hidden_state.squeeze(0)
    embedding = hidden_states.mean(dim=0).cpu().numpy()
    return embedding.astype("float32").reshape(1, -1)

def retrieve_chunks(query_text, top_k=10, distance_threshold=15):
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = embed_query(query_text)
    distances, indices = index.search(query_embedding, top_k)

    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        if float(dist) <= distance_threshold:  # ✅ filter here
            chunk = metadata[idx]
            retrieved.append({
                "distance": float(dist),
                "document_pdf": chunk["pdf_path"],
                "chunk_id": chunk["chunk_id"],
                "snippet": chunk["text"]
            })

    print(f"Filtered chunks: {len(retrieved)} under distance <= {distance_threshold}")
    return retrieved


def build_prompt(query, retrieved_chunks):
    prompt = "Use the following excerpts from PDF documents to answer the question as accurately as possible.\n\n"
    prompt += "Excerpts:\n"
    for item in retrieved_chunks:
        prompt += f"- From {item['document_pdf']} (chunk {item['chunk_id']}): \"{item['snippet']}\"\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    return prompt

# def generate_answer(prompt, max_tokens=200):
#     output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
#     return output.strip()




def generate_answer(prompt, max_tokens=200):
    
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{prompt}"
    )
    return response.text.strip()