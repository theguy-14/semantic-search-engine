from fastapi import FastAPI
from scripts.search import run_query
import os

if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

app = FastAPI()

@app.get("/query/{query}")
def query(query: str):
    response = run_query(query)
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the DocuFind API! Use /query/{query} to search."}
