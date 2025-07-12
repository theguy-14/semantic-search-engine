import os
import fitz  # PyMuPDF
import json
import nltk
from tqdm import tqdm

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

PDF_DIR = "./pdfs"
OUTPUT_JSON = "./data/all_chunks.json"
WORDS_PER_CHUNK = 500


def extract_structured_text(pdf_path):
    """Extract text with structure awareness from PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        # Extract with structure awareness
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] == 0:  # Text block type
                text += block[4] + "\n"
    doc.close()
    return text


def chunk_text(text, words_per_chunk=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        word_count = len(sentence.split())
        if current_word_count + word_count <= words_per_chunk:
            current_chunk += " " + sentence
            current_word_count += word_count
        else:
            if word_count > words_per_chunk:
                # Break up long sentence
                words = sentence.split()
                for i in range(0, len(words), words_per_chunk):
                    sub_chunk = " ".join(words[i:i+words_per_chunk])
                    chunks.append(sub_chunk)
                current_chunk = ""
                current_word_count = 0
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = word_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_all_pdfs(pdf_dir, output_path):
    all_chunks = []
    count = 0
    for filename in tqdm(os.listdir(pdf_dir)):
        count += 1
        if count > 10:
            continue
        
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            doc_title = os.path.splitext(filename)[0]

            try:
                # Use the improved structured text extraction
                text = extract_structured_text(pdf_path)
                chunks = chunk_text(text, words_per_chunk=WORDS_PER_CHUNK)

                for idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        "chunk_id": f"{doc_title}_chunk_{idx+1}",
                        "document": doc_title,
                        "text": chunk
                    })
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(all_chunks)} chunks from PDFs")


if __name__ == "__main__":
    process_all_pdfs(PDF_DIR, OUTPUT_JSON)