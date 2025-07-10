# run_query.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from retrieve_db import retrieve_chunks, build_prompt, generate_answer

if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or type 'exit' to quit'): ")
        if query.lower() == "exit":
            break

        print("\n🔍 Retrieving relevant chunks...")
        chunks = retrieve_chunks(query)

        print("📝 Building prompt & generating answer...")
        prompt = build_prompt(query, chunks)
        answer = generate_answer(prompt)

        output = {
            "question": query,
            "answer": answer,
            "sources": [
                {
                    "document_pdf": item["document_pdf"],
                    "chunk_id": item["chunk_id"],
                    "snippet": item["snippet"][:300]+"..."
                }
                for item in chunks
            ]
        }

        # print(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n📄 Answer: {output['answer']}")
