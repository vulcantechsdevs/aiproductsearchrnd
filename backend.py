# backend2.py
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import json
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = chromadb.PersistentClient(path="./chroma_db") 
text_collection = client.get_or_create_collection("products")  

image_collection = client.get_or_create_collection("products_image")

clip_model = SentenceTransformer("clip-ViT-B-32")

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
            return json.loads(specs)
        except:
            return []
    if isinstance(specs, list):
        return specs
    return []

def extract_product_id(collection_id):
    """
    Extract the actual product ID from collection IDs.
    - Text collection: "text-{UUID}" -> "{UUID}" 
    - Image collection: "image-{UUID}-{index}" -> "{UUID}"
    - Direct ID: "{UUID}" -> "{UUID}"
    """
    if collection_id.startswith("text-"):
        return collection_id[5:]  # Remove "text-" prefix
    elif collection_id.startswith("image-"):
        # For image format: "image-{UUID}-{index}"
        # Remove "image-" prefix first
        without_prefix = collection_id[6:]  # Remove "image-"
        # Find the last hyphen (before the index) and remove everything after it
        last_hyphen_index = without_prefix.rfind("-")
        if last_hyphen_index != -1:
            return without_prefix[:last_hyphen_index]  # Return UUID part
        return without_prefix  # Fallback if no index found
    else:
        return collection_id  # Already a clean product ID

def merge_meta(priority_meta, fallback_meta):
    """Merge fields from fallback_meta when missing in priority_meta."""
    if not priority_meta:
        priority_meta = {}
    if not fallback_meta:
        fallback_meta = {}

    merged = dict(fallback_meta)
    merged.update({k: v for k, v in priority_meta.items() if v not in (None, "", [], {})})
    return merged

def build_response(ids, metas, docs, distances):
    response = []
    for i, (pid, meta, doc, distance) in enumerate(zip(ids, metas, docs, distances)):
        description = (meta or {}).get("description", "")
        if not description and doc:
            parts = doc.split(": ", 1)
            description = parts[1] if len(parts) > 1 else doc

        # Extract actual product ID from image collection ID format
        actual_product_id = extract_product_id(pid)

        response.append({
            "id": actual_product_id,
            "name": (meta or {}).get("name", ""),
            "description": description,
            "images": parse_images(meta or {}),
            "specifications": parse_specs(meta or {}),
            "similarity_score": round(1 - distance, 3),
            "rank": i + 1
        })
    return response

def hydrate_with_text_metadata(ids, metas, docs):
    """
    For each result id, if images/name/description/specs are missing in metas,
    fetch from the text collection ('products') and merge.
    """
    hydrated_metas, hydrated_docs = [], []
    for pid, meta, doc in zip(ids, metas, docs):
        # Extract the actual product ID
        product_id = extract_product_id(pid)
        
        # Try both formats: just the ID and "text-{ID}"
        fallback_ids = [product_id, f"text-{product_id}"]
        
        # Try to find the product in the text collection
        fallback = None
        for fid in fallback_ids:
            try:
                fallback = text_collection.get(ids=[fid])
                if fallback and fallback.get("metadatas") and fallback["metadatas"]:
                    break
            except:
                continue
        
        if fallback and fallback.get("metadatas") and fallback["metadatas"]:
            fb_meta = fallback["metadatas"][0] or {}
            fb_doc = (fallback["documents"][0] or "") if fallback.get("documents") else ""
            merged_meta = merge_meta(meta or {}, fb_meta)
            description = merged_meta.get("description", "")
            if not description and (doc or fb_doc):
                parts = (doc or fb_doc).split(": ", 1)
                merged_meta["description"] = parts[1] if len(parts) > 1 else (doc or fb_doc)
        else:
            merged_meta = meta or {}
        
        hydrated_metas.append(merged_meta)
        hydrated_docs.append(doc)
    return hydrated_metas, hydrated_docs

# ---------- Routes ----------
@app.get("/products")
def get_products():
    """Return all products from the text collection"""
    items = text_collection.get()
    results = []
    for i, m, doc in zip(items.get("ids", []), items.get("metadatas", []), items.get("documents", [])):
        description = (m or {}).get("description", "")
        if not description and doc:
            parts = doc.split(": ", 1)
            description = parts[1] if len(parts) > 1 else doc

        results.append({
            "id": i,
            "name": (m or {}).get("name", ""),
            "description": description,
            "images": parse_images(m or {}),
            "specifications": parse_specs(m or {})
        })
    return results

@app.get("/search")
def search_products(q: str = Query(...), n: int = 5):
    """Semantic search by text against the text collection"""
    results = text_collection.query(query_texts=[q], n_results=n)
    return build_response(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0]
    )

@app.post("/image-search")
async def image_search(file: UploadFile = File(...), n: int = 5):
    """Semantic search by image using CLIP → then hydrate from text metadata"""
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Encode as batch to be safe
    embedding = clip_model.encode([img], convert_to_numpy=True)[0].tolist()

    results = image_collection.query(query_embeddings=[embedding], n_results=n)

    # Hydrate: pull full product metadata (name, images, specs, description) from text collection
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]

    metas_h, docs_h = hydrate_with_text_metadata(ids, metas, docs)

    return build_response(ids, metas_h, docs_h, distances)