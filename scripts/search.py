# run_query.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from collections import defaultdict
from os.path import basename
from scripts.retrieve_db import retrieve_chunks, build_prompt, generate_answer

with open("./data/pdf_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# if __name__ == "__main__":
#     while True:
#         query = input("\nEnter your query (or type 'exit' to quit'): ")
#         if query.lower() == "exit":
#             break

#         print("\n🔍 Retrieving relevant chunks...")
def run_query(query):
        chunks = retrieve_chunks(query)

        print("\n🔍 Grouping chunks by document...")
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
        # print(f"\n📄 Answer: {answer}")

        print(prompt)

        results = []

        for doc in doc_objects:
            prompt = (
                f"answer the question as precisely as possible.\n\n"
                f"Content:\n{doc['concatenated_text']}\n\n"
                f"Question: {query}\nAnswer:"
            )

            object_key = basename(f"{doc['doc_name']}").split("v")[0]
            data = metadata.get(object_key)

            answer = generate_answer(prompt) 
            results.append({
                "title": data.get("title"),
                "authors": data.get("authors"),
                "doc_id": doc["doc_id"],
                "doc_name": basename(f"{doc['doc_name']}"),
                "source": doc["doc_name"],
                "context": answer
            })

            documents=[]
            for result in results:
                documents.append(result["doc_name"])

        final_response = {
            "question": query,
            "overall_response": answer,
            "documents": documents,
            "docwise_responses": results
        }

        return final_response