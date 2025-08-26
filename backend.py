# backend.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import json

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. Restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to chroma db
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("products")

# ---------- Helpers ----------
def parse_images(meta):
    images = meta.get("images")
    if not images:
        return []
    if isinstance(images, str):
        return [img.strip() for img in images.split(",") if img.strip()]
    if isinstance(images, list):
        return images
    return []

def parse_specs(meta):
    specs = meta.get("specifications")
    if not specs:
        return []
    if isinstance(specs, str):
        try:
            return json.loads(specs)  # parse JSON string into list
        except:
            return []
    if isinstance(specs, list):
        return specs
    return []

# ---------- Routes ----------
@app.get("/products")
def get_products():
    """Return all products"""
    items = collection.get()
    results = []
    for i, m, doc in zip(items["ids"], items["metadatas"], items["documents"]):
        description = m.get("description", "")
        if not description and doc:
            parts = doc.split(": ", 1)
            description = parts[1] if len(parts) > 1 else doc

        results.append({
            "id": i,
            "name": m.get("name", ""),
            "description": description,
            "images": parse_images(m),
            "specifications": parse_specs(m)
        })
    return results

@app.get("/search")
def search_products(q: str = Query(...)):
    """Semantic search by query"""
    results = collection.query(query_texts=[q])
    response = []
    for i, (ids, meta, doc, distance) in enumerate(zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0]
    )):
        description = meta.get("description", "")
        if not description and doc:
            parts = doc.split(": ", 1)
            description = parts[1] if len(parts) > 1 else doc

        response.append({
            "id": ids,
            "name": meta.get("name", ""),
            "description": description,
            "images": parse_images(meta),
            "specifications": parse_specs(meta),
            "similarity_score": round(1 - distance, 3),
            "rank": i + 1
        })
    return response
