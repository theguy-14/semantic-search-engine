# run_query.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from collections import defaultdict
from os.path import basename
from scripts.fixed_search import FixedSearchEngine

with open("./data/pdf_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def run_query(query):
    """Run query using the fixed search system"""
    # Initialize the fixed search engine
    engine = FixedSearchEngine()
    
    # Run the query with improved retrieval
    result = engine.run_fixed_query(query)
    
    # Process results to match expected format
    doc_objects = []
    for doc_response in result.get('docwise_responses', []):
        doc_objects.append({
            "doc_id": doc_response.get('doc_id', ''),
            "doc_name": doc_response.get('doc_name', ''),
            "concatenated_text": doc_response.get('context', '')
        })
    
    # Build final response
    results = []
    for doc in doc_objects:
        object_key = basename(f"{doc['doc_name']}").split("v")[0]
        data = metadata.get(object_key, {})
        
        results.append({
            "title": data.get("title", "Unknown"),
            "authors": data.get("authors", []),
            "doc_id": doc["doc_id"],
            "doc_name": basename(f"{doc['doc_name']}"),
            "source": doc["doc_name"],
            "context": doc['concatenated_text']
        })
    
    documents = [result["doc_name"] for result in results]
    
    final_response = {
        "question": query,
        "overall_response": result.get("answer", "No answer generated"),
        "documents": documents,
        "docwise_responses": results
    }
    
    return final_response