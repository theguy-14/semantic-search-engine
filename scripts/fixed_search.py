#!/usr/bin/env python3
"""
Fixed semantic search system - Query functionality only
"""

import json
import os
import torch
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from typing import List, Dict, Any
from collections import defaultdict

# Load environment variables
load_dotenv()

class FixedSearchEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use better embedding model for semantic search
        self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.index_path = "faiss_index/fixed_chunks.index"
        self.metadata_path = "data/fixed_metadata.json"
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load improved embedding and reranking models"""
        print("Loading improved embedding model...")
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.embed_model.to(self.device)
        
        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed query using improved model"""
        embedding = self.embed_model.encode([text])
        return embedding.astype("float32")
    
    def retrieve_chunks(self, query_text: str, top_k: int = 10, use_reranking: bool = True) -> List[Dict]:
        """Retrieve chunks with improved strategy"""
        # Load index and metadata if not already loaded
        if not hasattr(self, 'index'):
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                print("❌ Index not found. Please run embed_and_index.py first to create the index.")
                return []
        
        # Get initial candidates (more for reranking)
        initial_k = top_k * 4 if use_reranking else top_k
        query_embedding = self.embed_query(query_text)
        
        # Search with fixed index
        distances, indices = self.index.search(query_embedding, initial_k)
        
        # Get candidates with quality filtering
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.metadata[idx]
            
            # Filter by quality score
            if chunk.get("quality_score", 0) >= 0.6:
                candidates.append({
                    "distance": float(dist),
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["document"],
                    "text": chunk["text"],
                    "pdf_path": chunk["pdf_path"],
                    "quality_score": chunk.get("quality_score", 0)
                })
        
        # Apply reranking if enabled
        if use_reranking and len(candidates) > 0:
            print("Applying cross-encoder reranking...")
            candidates = self.rerank_chunks(query_text, candidates)
        
        return candidates[:top_k]
    
    def rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank chunks using cross-encoder"""
        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.reranker.predict(pairs)
        
        # Sort by reranking scores
        reranked = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
        return reranked
    
    def build_fixed_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Build improved prompt for Gemini"""
        prompt = "You are a helpful assistant that answers questions based on the provided document excerpts. "
        prompt += "Please answer the question accurately using only the information from the excerpts below. "
        prompt += "If the excerpts don't contain relevant information, say so clearly.\n\n"
        
        prompt += "Document Excerpts:\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            prompt += f"{i}. From {chunk['document']}:\n"
            prompt += f"   \"{chunk['text']}\"\n\n"
        
        prompt += f"Question: {query}\n\n"
        prompt += "Answer: "
        
        return prompt
    
    def generate_answer_gemini(self, prompt: str, max_tokens: int = 600) -> str:
        """Generate answer using Gemini with better configuration"""
        if not self.gemini_api_key:
            return "Error: GEMINI_API_KEY not found"
        
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.2,  # Very low temperature for focused answers
                    "topP": 0.7,
                    "topK": 20
                }
            }
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0]["text"].strip()
            
            return "No response generated from the model."
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run_fixed_query(self, query: str) -> Dict[str, Any]:
        """Run fixed query with better retrieval and generation"""
        print(f"🔍 Processing query: {query}")
        
        # Retrieve chunks with fixed strategy
        chunks = self.retrieve_chunks(query, top_k=6, use_reranking=True)
        
        if not chunks:
            return {
                "question": query,
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "documents": []
            }
        
        # Group by document
        grouped_docs = defaultdict(list)
        for chunk in chunks:
            grouped_docs[chunk['document']].append(chunk['text'])
        
        # Build fixed prompt
        prompt = self.build_fixed_prompt(query, chunks)
        
        # Generate answer
        answer = self.generate_answer_gemini(prompt)
        
        # Prepare response
        doc_objects = []
        for idx, (doc_name, snippets) in enumerate(grouped_docs.items()):
            doc_objects.append({
                "doc_id": f"doc_{idx+1}",
                "doc_name": doc_name,
                "context": " ".join(snippets)
            })
        
        return {
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "document": chunk["document"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "quality_score": chunk.get("quality_score", 0)
                }
                for chunk in chunks
            ],
            "documents": [doc["doc_name"] for doc in doc_objects],
            "docwise_responses": doc_objects
        } 