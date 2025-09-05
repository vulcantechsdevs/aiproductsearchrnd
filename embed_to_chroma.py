"""
embed_to_chroma_batch.py
- Batch process products from Postgres
- Embeds product text with all-MiniLM-L6-v2
- Embeds product images with clip-ViT-B-32
- Stores text embeddings in collection "products_text"
- Stores image embeddings in collection "products_image"
"""

import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
import json
import requests
from PIL import Image
from io import BytesIO
import time

# -------- Config ----------
BATCH_SIZE = 500          # fetch & embed per batch
MAX_PRODUCTS = 200_000    # adjust if needed

# -------- Postgres connection ----------
conn = psycopg2.connect(
    dbname="medworld",
    user="postgres",
    password="1",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# -------- Models ----------
print("Loading models...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")   # text
clip_model = SentenceTransformer("clip-ViT-B-32")      # image (CLIP)
print("Models loaded.")

# -------- Chroma client ----------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or reuse collections
text_collection = chroma_client.get_or_create_collection(
    name="products_text",
    metadata={"hnsw:space": "cosine"}
)
image_collection = chroma_client.get_or_create_collection(
    name="products_image",
    metadata={"hnsw:space": "cosine"}
)

# -------- Helper functions ----------
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

# -------- Batch Processing ----------
offset = 0
total_inserted = 0

while True:
    cur.execute(
        "SELECT id, oem_id, name, description, images, specifications "
        "FROM products.products_info "
        "ORDER BY id ASC "
        f"LIMIT {BATCH_SIZE} OFFSET {offset};"
    )
    rows = cur.fetchall()
    if not rows:
        break

    print(f"\n📦 Processing batch offset={offset}, size={len(rows)}")

    # ----- TEXT embeddings -----
    texts = []
    text_ids = []
    text_metas = []
    for prod_id, oem_id, name, description, images_field, specifications in rows:
        prod_id = str(prod_id)
        oem_id = str(oem_id) if oem_id is not None else ""   # handle NULLs
        name = name or ""
        description = description or ""
        specs_str = specs_to_string(specifications)
        image_list = normalize_image_list(images_field)
        images_str = ",".join(image_list)

        content = f"{name}. {description}. Specs: {specs_str}"

        texts.append(content)
        text_ids.append(f"text-{prod_id}")
        text_metas.append({
            "id": prod_id,
            "oem_id": oem_id,   # ✅ added here
            "type": "text",
            "name": name,
            "description": description,
            "images": images_str,
            "specifications": specs_str
        })

    if texts:
        text_embs = text_model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
        text_collection.add(ids=text_ids, documents=texts, embeddings=text_embs, metadatas=text_metas)
        total_inserted += len(texts)
        print(f"✅ Inserted {len(texts)} text embeddings (total={total_inserted})")

    # ----- IMAGE embeddings -----
    for prod_id, oem_id, name, description, images_field, specifications in rows:
        prod_id = str(prod_id)
        oem_id = str(oem_id) if oem_id is not None else ""   # handle NULLs
        specs_str = specs_to_string(specifications)
        image_list = normalize_image_list(images_field)
        images_str = ",".join(image_list)

        for idx, img_url in enumerate(image_list):
            try:
                resp = requests.get(img_url, timeout=6)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img_emb = clip_model.encode(img, normalize_embeddings=True).tolist()

                image_collection.add(
                    ids=[f"image-{prod_id}-{idx}"],
                    documents=[f"{name} (image)"],
                    embeddings=[img_emb],
                    metadatas=[{
                        "id": prod_id,
                        "oem_id": oem_id,   # ✅ added here
                        "type": "image",
                        "name": name,
                        "description": description or "",
                        "images": images_str,
                        "specifications": specs_str
                    }]
                )
                print(f"   🖼️ image embedded {prod_id}-{idx}")
            except Exception as e:
                print(f"⚠️ image failed for {prod_id} url={img_url}: {e}")
            time.sleep(0.05)  # avoid hammering servers

    offset += BATCH_SIZE
    if offset >= MAX_PRODUCTS:
        break

print(f"\n🎉 Done. Inserted ~{total_inserted} text products into ChromaDB.")
cur.close()
conn.close()
