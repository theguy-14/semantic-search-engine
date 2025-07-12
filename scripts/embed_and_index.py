#!/usr/bin/env python3
"""
Improved embedding and indexing system
"""

import json
import os
import torch
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict
import gc

# Paths
CHUNK_PATH = "data/all_chunks.json"
INDEX_PATH = "faiss_index/.indfixed_chunksex"
METADATA_PATH = "data/fixed_metadata.json"

class ImprovedEmbeddingIndexer:
    def __init__(self):
        # Set environment variables to avoid multiprocessing issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use better embedding model for semantic search
        self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Load model with memory optimization
        print("Loading improved embedding model...")
        try:
            self.embed_model = SentenceTransformer(self.embed_model_name, device=self.device)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
    def create_fixed_index(self):
        """Create a fixed FAISS index with better chunking and embedding"""
        print("Creating fixed index with better strategy...")
        
        try:
            # Load existing chunks
            with open(CHUNK_PATH, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            print(f"Loaded {len(chunks_data)} chunks from {CHUNK_PATH}")
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return None, None
        
        # Apply better chunking strategy
        fixed_chunks = self.fix_chunks(chunks_data)
        
        if not fixed_chunks:
            print("❌ No valid chunks found after filtering")
            return None, None
        
        print(f"✅ Filtered to {len(fixed_chunks)} high-quality chunks")
        
        # Generate embeddings with better model
        texts = [chunk["text"] for chunk in fixed_chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        try:
            # Use smaller batch size to avoid memory issues
            batch_size = 16 if self.device.type == "cpu" else 32
            embeddings = self.embed_model.encode(
                texts, 
                show_progress_bar=True, 
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print("✅ Embeddings generated successfully")
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return None, None
        
        # Clear GPU memory if using CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Normalize embeddings (already done by sentence-transformers)
        # faiss.normalize_L2(embeddings)  # Not needed as normalize_embeddings=True
        
        # Create better FAISS index
        dimension = embeddings.shape[1]
        print(f"Embedding dimension: {dimension}")
        
        # Use simpler index for better stability
        index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        index.add(embeddings.astype("float32"))
        
        # Save fixed index and metadata
        try:
            os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
            faiss.write_index(index, INDEX_PATH)
            print(f"✅ Index saved to {INDEX_PATH}")
            
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(fixed_chunks, f, indent=2, ensure_ascii=False)
            print(f"✅ Metadata saved to {METADATA_PATH}")
        except Exception as e:
            print(f"❌ Error saving index/metadata: {e}")
            return None, None
        
        print(f"✅ Fixed index created with {len(fixed_chunks)} chunks")
        return index, fixed_chunks
    
    def fix_chunks(self, chunks_data: List[Dict]) -> List[Dict]:
        """Apply better chunking strategy"""
        fixed_chunks = []
        
        print("Filtering and cleaning chunks...")
        
        # Diagnostic counters
        total_chunks = len(chunks_data)
        quality_rejected = 0
        cleaning_rejected = 0
        accepted = 0
        
        for chunk in tqdm(chunks_data, desc="Processing chunks"):
            text = chunk["text"]
            
            # Skip chunks with poor quality (relaxed threshold)
            quality_score = self.calculate_text_quality(text)
            if quality_score < 0.3:  # Lowered from 0.5
                quality_rejected += 1
                continue
            
            # Clean and improve text
            improved_text = self.clean_text(text)
            
            # Accept if cleaning didn't remove too much content
            if len(improved_text.split()) >= 20:  # Lowered threshold since chunks are uniform
                fixed_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["document"],
                    "text": improved_text,
                    "pdf_path": chunk.get("pdf_path", f"pdfs/{chunk['document']}.pdf"),
                    "length": len(improved_text.split()),
                    "quality_score": quality_score
                })
                accepted += 1
            else:
                cleaning_rejected += 1
        
        # Print diagnostic information
        print(f"\n📊 Filtering Statistics:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Quality rejected: {quality_rejected}")
        print(f"  Cleaning rejected: {cleaning_rejected}")
        print(f"  Accepted: {accepted}")
        
        return fixed_chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and improve text quality"""
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove excessive punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
        
        # Remove excessive dashes
        text = re.sub(r'-{2,}', '-', text)
        
        # Remove arXiv headers and metadata
        text = re.sub(r'arXiv:\d+\.\d+v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d+', '', text)
        
        # Remove excessive numbers and symbols
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence endings
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()
    
    def calculate_text_quality(self, text: str) -> float:
        """Calculate quality score for text"""
        score = 0.0
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) >= 3:
            score += 0.3
        
        # Content quality score
        common_words = ['the', 'and', 'or', 'in', 'on', 'of', 'to', 'for', 'with', 'by', 'from']
        if any(word in text.lower() for word in common_words):
            score += 0.3
        
        # No excessive caps
        if not text.isupper():
            score += 0.2
        
        # No excessive numbers/symbols
        if len(re.findall(r'[^\w\s]', text)) < len(text) * 0.15:
            score += 0.2
        
        return score

def main():
    """Create the improved index"""
    print("🚀 Creating Improved Embedding Index")
    print("=" * 50)
    
    try:
        indexer = ImprovedEmbeddingIndexer()
        result = indexer.create_fixed_index()
        
        if result[0] is not None:
            print("✅ Index creation completed successfully!")
        else:
            print("❌ Index creation failed!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()