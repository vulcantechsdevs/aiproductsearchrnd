from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException
import io
import json
import requests
from io import BytesIO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Chroma client & collections ----------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
text_collection = chroma_client.get_collection("products_text")
image_collection = chroma_client.get_collection("products_image")

# ---------- Models ----------
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = SentenceTransformer("clip-ViT-B-32")

# ---------- Helpers ----------
def parse_images_from_meta(meta):
    img_str = meta.get("images") if meta else ""
    if not img_str:
        return []
    return [s.strip() for s in str(img_str).split(",") if s.strip()]

def build_result_from_meta(meta, doc=None, distance=None, rank=None):
    if meta.get("deleted"):  # skip deleted
        return None
    return {
        "id": meta.get("id"),
        "oem_id": meta.get("oem_id"),
        "name": meta.get("name"),
        "description": meta.get("description") or (doc or ""),
        "images": parse_images_from_meta(meta),
        "specifications": meta.get("specifications"),
        "similarity_score": round(1 - distance, 3) if distance is not None else None,
        "rank": rank,
    }

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Product search (text + image) backend running."}

# ---- All products with pagination ----
@app.get("/products")
def get_all_products(offset: int = 0, limit: int = 50):
    results = text_collection.get(include=["documents", "metadatas"], offset=offset, limit=limit)

    products = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        item = build_result_from_meta(meta, doc)
        if item:
            products.append(item)

    return {
        "results": products,
        "offset": offset,
        "limit": limit,
        "count": len(products),
    }

# ---- Text search ----
@app.get("/search")
def search_text(q: str = Query(...), top_k: int = 100):
    q = q.strip()
    if not q:
        return {"results": []}

    q_emb = text_model.encode(q, normalize_embeddings=True).tolist()
    results = text_collection.query(query_embeddings=[q_emb], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for i, (m, doc, d) in enumerate(zip(metas, docs, dists)):
        item = build_result_from_meta(m, doc, distance=d, rank=i + 1)
        if item:
            out.append(item)
    return {"results": out}

# ---- Image search ----
@app.post("/image-search")
async def search_image(file: UploadFile = File(...), top_k: int = 100):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image uploaded: {e}"}

    q_emb = clip_model.encode(img, normalize_embeddings=True).tolist()
    results = image_collection.query(query_embeddings=[q_emb], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for i, (m, doc, d) in enumerate(zip(metas, docs, dists)):
        if m.get("deleted"):  # skip deleted
            continue

        prod_id = m.get("id")
        hydrated_meta = dict(m)

        try:
            text_res = text_collection.get(ids=[f"text-{prod_id}"])
            if text_res and text_res.get("metadatas") and text_res["metadatas"][0]:
                tmeta = text_res["metadatas"][0][0] if isinstance(text_res["metadatas"][0], list) else text_res["metadatas"][0]
                hydrated_meta.update({k: tmeta.get(k) for k in ("name", "description", "images", "specifications", "id", "oem_id", "deleted")})
        except Exception:
            pass

        item = build_result_from_meta(hydrated_meta, doc, distance=d, rank=i + 1)
        if item:
            out.append(item)

    return {"results": out}

class ProductPayload(BaseModel):
    id: str
    oem_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    images: Optional[str] = None   # comma-separated URLs
    specifications: Optional[str] = None

    # ---- Insert product ----
def normalize_image_list(images_field):
    if not images_field:
        return []
    if isinstance(images_field, list):
        return [str(x).strip() for x in images_field if str(x).strip()]
    s = str(images_field).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return [p.strip() for p in s.split(",") if p.strip()]

def specs_to_string(spec_field):
    if not spec_field:
        return ""
    if isinstance(spec_field, str):
        try:
            parsed = json.loads(spec_field)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return spec_field
    try:
        return json.dumps(spec_field, ensure_ascii=False)
    except Exception:
        return str(spec_field)
# -----New product insert------
@app.post("/insert")
def insert_product(payload: dict):
    prod_id = payload.get("id")
    if not prod_id:
        return {"error": "Product id is required"}

    existing = text_collection.get(ids=[f"text-{prod_id}"])
    if existing and existing.get("metadatas") and existing["metadatas"][0] and not existing["metadatas"][0].get("deleted", False):
        return {"message": f"Product {prod_id} already exists"}

    meta = {
        "id": prod_id,
        "name": payload.get("name"),
        "description": payload.get("description"),
        "images": payload.get("images"),
        "specifications": payload.get("specifications"),
        "deleted": False
    }

    doc_text = meta.get("description") or meta.get("name") or ""
    emb = text_model.encode(doc_text, normalize_embeddings=True).tolist()

    text_collection.upsert(
        ids=[f"text-{prod_id}"],
        documents=[doc_text],
        metadatas=[meta],
        embeddings=[emb],
    )

    return {"message": f"Product {prod_id} inserted successfully"}


# ---- Update product ----
@app.post("/update")
def update_product(payload: dict):
    prod_id = payload.get("id")
    if not prod_id:
        return {"error": "Product id is required"}

    existing = text_collection.get(ids=[f"text-{prod_id}"])
    if not existing or not existing.get("metadatas") or not existing["metadatas"][0]:
        return {"error": f"Product {prod_id} not found"}

    meta = dict(existing["metadatas"][0])
    meta.update(payload)
    meta["deleted"] = False  # keep active if updated

    doc_text = meta.get("description") or meta.get("name") or ""
    new_emb = text_model.encode(doc_text, normalize_embeddings=True).tolist()

    text_collection.upsert(
        ids=[f"text-{prod_id}"],
        documents=[doc_text],
        metadatas=[meta],
        embeddings=[new_emb],
    )

    return {"message": f"Product {prod_id} updated successfully", "updated_meta": meta}

# ---- Permanent delete product ----
@app.post("/delete")
def permanent_delete_product(payload: dict):
    prod_id = payload.get("id")
    if not prod_id: 
        return {"error": "Product id is required"}
    
    existing = text_collection.get(ids=[f"text-{prod_id}"])
    if not existing or not existing.get("metadatas") or not existing["metadatas"][0]:
        return {"error": f"Product {prod_id} not found"}

    # Delete from collection
    text_collection.delete(ids=[f"text-{prod_id}"])

    return {"message": f"Product {prod_id}  deleted successfully"}

# -----New product insert------
# @app.post("/insert")
# def insert_product(payload: dict):
#     oem_id = payload.get("oem_id")
#     if not oem_id:
#         return {"error": "OEM ID is required"}

#     existing = text_collection.get(ids=[f"text-{oem_id}"])
#     if existing and existing.get("metadatas") and existing["metadatas"][0] and not existing["metadatas"][0].get("deleted", False):
#         return {"message": f"Product with OEM ID {oem_id} already exists"}

#     meta = {
#         "id": payload.get("id"),   # keep original DB ID in metadata
#         "oem_id": oem_id,
#         "name": payload.get("name"),
#         "description": payload.get("description"),
#         "images": payload.get("images"),
#         "specifications": payload.get("specifications"),
#         "deleted": False
#     }

#     doc_text = meta.get("description") or meta.get("name") or ""
#     emb = text_model.encode(doc_text, normalize_embeddings=True).tolist()

#     text_collection.upsert(
#         ids=[f"text-{oem_id}"],
#         documents=[doc_text],
#         metadatas=[meta],
#         embeddings=[emb],
#     )

#     return {"message": f"Product with OEM ID {oem_id} inserted successfully"}


# # ---- Update product ----
# @app.post("/update")
# def update_product(payload: dict):
#     oem_id = payload.get("oem_id")
#     if not oem_id:
#         return {"error": "OEM ID is required"}

#     existing = text_collection.get(ids=[f"text-{oem_id}"])
#     if not existing or not existing.get("metadatas") or not existing["metadatas"][0]:
#         return {"error": f"Product with OEM ID {oem_id} not found"}

#     meta = dict(existing["metadatas"][0])
#     meta.update(payload)
#     meta["deleted"] = False  # keep active if updated

#     doc_text = meta.get("description") or meta.get("name") or ""
#     new_emb = text_model.encode(doc_text, normalize_embeddings=True).tolist()

#     text_collection.upsert(
#         ids=[f"text-{oem_id}"],
#         documents=[doc_text],
#         metadatas=[meta],
#         embeddings=[new_emb],
#     )

#     return {"message": f"Product with OEM ID {oem_id} updated successfully", "updated_meta": meta}


# # ---- Permanent delete product ----
# @app.post("/delete")
# def permanent_delete_product(payload: dict):
#     oem_id = payload.get("oem_id")
#     if not oem_id:
#         return {"error": "OEM ID is required"}
    
#     existing = text_collection.get(ids=[f"text-{oem_id}"])
#     if not existing or not existing.get("metadatas") or not existing["metadatas"][0]:
#         return {"error": f"Product with OEM ID {oem_id} not found"}

#     # Delete from collection
#     text_collection.delete(ids=[f"text-{oem_id}"])

#     return {"message": f"Product with OEM ID {oem_id} deleted successfully"}