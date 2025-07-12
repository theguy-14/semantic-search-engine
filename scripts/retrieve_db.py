# retrieval_engine.py

import json
import torch
import faiss
import numpy as np
import requests
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# -------------------------------
# CONFIG
# -------------------------------
INDEX_PATH = "faiss_index/chunks.index"
METADATA_PATH = "data/metadata.json"
EMBED_MODEL_NAME = "allenai/longformer-base-4096"
GEN_LLM_MODEL = "EleutherAI/gpt-neo-125M"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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
gen_model = AutoModelForCausalLM.from_pretrained(GEN_LLM_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
gen_model.to(device)
generator = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer, device=0 if torch.cuda.is_available() else -1)

print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------
# FUNCTIONS TO USE ANYWHERE
# -------------------------------

def embed_query(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = embed_model(**inputs)
    hidden_states = output.last_hidden_state.squeeze(0)
    embedding = hidden_states.mean(dim=0).cpu().numpy()
    return embedding.astype("float32").reshape(1, -1)

def rerank_chunks(query, chunks):
    """Rerank chunks using cross-encoder for better relevance"""
    pairs = [(query, chunk['snippet']) for chunk in chunks]
    scores = reranker.predict(pairs)
    reranked = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return reranked[:10]  # Return top 10 reranked

def retrieve_chunks(query_text, top_k=5, use_reranking=True):
    """Retrieve and optionally rerank chunks"""
    # Get initial candidates (more than final top_k for reranking)
    initial_k = top_k * 2 if use_reranking else top_k
    query_embedding = embed_query(query_text)
    distances, indices = index.search(query_embedding, initial_k)
    
    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk =  metadata[idx]
        retrieved.append({
            "distance": float(dist),
            "document_pdf": chunk["pdf_path"],
            "chunk_id": chunk["chunk_id"],
            "snippet": chunk["text"]
        })
    
    # Apply reranking if enabled
    if use_reranking:
        print("Reranking chunks for better relevance...")
        retrieved = rerank_chunks(query_text, retrieved)
    
    return retrieved[:top_k]

def build_prompt(query, retrieved_chunks):
    prompt = "Use the following excerpts from PDF documents to answer the question as accurately as possible.\n\n"
    prompt += "Excerpts:\n"
    for item in retrieved_chunks:
        prompt += f"- From {item['document_pdf']} (chunk {item['chunk_id']}): \"{item['snippet']}\"\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    return prompt

def generate_answer_gemini(prompt, max_tokens=500):
    """Generate answer using Google Gemini API"""
    try:
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        # Add API key as query parameter
        url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract response from Gemini API format
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"].strip()
        
        return "No response generated from the model."
            
    except requests.exceptions.RequestException as e:
        print(f"Gemini API request failed: {e}")
        return f"Error: Failed to generate response. {str(e)}"
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        return "Error: Unexpected response format from API."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error: {str(e)}"

def generate_answer_local(prompt, max_tokens=200):
    """Fallback to local model if Gemini fails"""
    try:
        output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)[0]["generated_text"]
        return output.strip()
    except Exception as e:
        return f"Local generation failed: {str(e)}"

def generate_answer(prompt, max_tokens=500, use_fallback=True):
    """Generate answer with Gemini as primary and local model as fallback"""
    # Try Gemini first
    answer = generate_answer_gemini(prompt, max_tokens)
    
    # If Gemini fails and fallback is enabled, use local model
    if answer.startswith("Error:") and use_fallback:
        print("Gemini failed, falling back to local model...")
        answer = generate_answer_local(prompt, max_tokens//2)
    
    return answer

# -------------------------------
# USAGE TRACKING AND CACHING
# -------------------------------

class ResponseCache:
    def __init__(self, cache_dir="cache", cache_duration_hours=24):
        self.cache_dir = cache_dir
        self.cache_duration_hours = cache_duration_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt):
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get_cached_response(self, prompt):
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid (simple time check)
                from datetime import datetime, timedelta
                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cached_time < timedelta(hours=self.cache_duration_hours):
                    return cache_data['response']
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        
        return None
    
    def cache_response(self, prompt, response):
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        from datetime import datetime
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'response': response
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

# Initialize cache
cache = ResponseCache()

def generate_answer_cached(prompt, max_tokens=500):
    """Generate answer with caching"""
    # Check cache first
    cached_response = cache.get_cached_response(prompt)
    if cached_response:
        print("Using cached response")
        return cached_response
    
    # Generate new response
    response = generate_answer(prompt, max_tokens)
    
    # Cache the response if it's not an error
    if not response.startswith("Error:"):
        cache.cache_response(prompt, response)
    
    return response