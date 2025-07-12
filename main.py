from fastapi import FastAPI, Body, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from scripts.search import run_query
from scripts.library import update_library_entry,get_docs_in_library
from scripts.pdf_chunker import process_all_pdfs
from scripts.embed_and_index import build_faiss_index
from scripts.metadata_fetcher import process_arxiv_pdfs
import os



if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/query/{query}")
def query(query: str):
    response = run_query(query)
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the DocuFind API! Use /query/{query} to search."}

@app.post("/update_library")
def update_library(data: dict = Body(...)):
    return update_library_entry(
        data.get("file_name"),
        data.get("doc_id"),
        data.get("library_name")
    )

@app.get("/library/{library_name}")
def get_docs(library_name: str):
    return get_docs_in_library(library_name)



@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        # Save files to disk or process them here
        with open(f"./pdfs/{file.filename}", "wb") as f:
            f.write(contents)
    return {"message": "Files uploaded successfully"}


@app.get("/process_pdfs")
async def process_pdfs_route():
    res=process_all_pdfs()
    print("PDFs processed and chunks saved.")
    return {"status": "PDFs processed", "res" :res}

@app.get("/create_index")
async def create_index_route():
    res=build_faiss_index()
    print("FAISS index created and metadata saved.")
    return {"status": "FAISS index created", "res": res}

@app.get("/fetch_metadata")
async def fetch_metadata_route():
    res= process_arxiv_pdfs()
    print("Metadata fetched and saved.")
    return {"status": "Metadata fetched", "res": res}

