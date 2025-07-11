import os
import requests
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

PDF_DIR = "pdfs"
OUTPUT_PATH = "data/pdf_metadata.json"

def extract_arxiv_id(filename):
    return filename.split("v")[0]

def get_arxiv_metadata(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    root = ET.fromstring(response.content)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        return None

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
    authors = [author.find("atom:name", ns).text.strip() for author in entry.findall("atom:author", ns)]
    published = entry.find("atom:published", ns).text.strip()
    pdf_url = None
    for link in entry.findall("atom:link", ns):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib["href"]
            break

    return {
        "title": title,
        "authors": authors,
        "release_date": published.split("T")[0],
        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    }

def process_arxiv_pdfs(pdf_dir):
    metadata_store = {}
    filenames = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for filename in tqdm(filenames):
        arxiv_id = extract_arxiv_id(filename)
        metadata = get_arxiv_metadata(arxiv_id)

        if metadata:
            metadata["file_name"] = filename
        else:
            metadata = {
                "title": "Not found",
                "authors": [],
                "release_date": "Not available",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "file_name": filename
            }

        metadata_store[arxiv_id] = metadata

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)

    print(f"\n Metadata saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_arxiv_pdfs(PDF_DIR)
