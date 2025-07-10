# run_query.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from collections import defaultdict
from retrieve_db import retrieve_chunks, build_prompt, generate_answer

if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or type 'exit' to quit'): ")
        if query.lower() == "exit":
            break

        print("\n🔍 Retrieving relevant chunks...")
        chunks = retrieve_chunks(query)

        

        grouped_docs = defaultdict(list)
        for item in chunks:
            grouped_docs[item['document_pdf']].append(item['snippet'])

        doc_objects = []
        for idx, (doc_name, snippets) in enumerate(grouped_docs.items()):
            concatenated_text = " ".join(snippets)
            doc_obj = {
                "doc_id": f"doc_{idx+1}",
                "doc_name": doc_name,
                "concatenated_text": concatenated_text
            }
            doc_objects.append(doc_obj)

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
                    "snippet": item["snippet"]
                }
                for item in chunks
            ]
        }

        # print(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n📄 Answer: {answer}")

        print(prompt)

        results = []

        # for doc in doc_objects:
        #     prompt = (
        #         f"Using the following excerpts from the document {doc['doc_name']}, "
        #         f"answer the question as precisely as possible.\n\n"
        #         f"Content:\n{doc['concatenated_text']}\n\n"
        #         f"Question: {query}\nAnswer:"
        #     )
        #     answer = generate_answer(prompt) 
        #     results.append({
        #         "doc_id": doc["doc_id"],
        #         "doc_name": doc["doc_name"],
        #         "answer": answer
        #     })

        #     print(results)