import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
import json
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
import time
import torch
import os
import shutil

# -------- Config ----------
BATCH_SIZE = 500         # Increased for faster processing (adjust based on memory)
MAX_PRODUCTS = 200_000    # Adjust if needed
IMAGE_BATCH_SIZE = 32     # Process images in batches for embedding
SLEEP_DELAY = 0.01        # Reduced delay for image downloads (adjust based on server)
DB_PATH = "./chroma_db"

# -------- Reset ChromaDB folder ----------
def clear_folder(folder):
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}: {e}")
    print(f"🗑️ Cleared contents of ChromaDB folder: {folder}")

# Clear the ChromaDB folder before starting
clear_folder(DB_PATH)

# -------- Postgres connection ----------
conn = psycopg2.connect(
    dbname="medworld",
    user="postgres",
    password="1",
    host="host.docker.internal",
    port="5432"
)
cur = conn.cursor()

# -------- Models ----------
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
text_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)   # Move to GPU if available
clip_model = SentenceTransformer("clip-ViT-B-32").to(device)      # Move to GPU if available
print("Models loaded.")

# -------- Chroma client ----------
chroma_client = chromadb.PersistentClient(path=DB_PATH)

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

async def fetch_image(session, url, timeout=6):
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.read()
            return None
    except Exception:
        return None

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
        oem_id = str(oem_id) if oem_id is not None else ""
        name = name or ""
        description = description or ""
        specs_str = specs_to_string(specifications)
        image_list = normalize_image_list(images_field)
        images_str = ",".join(image_list)

        # Include id and oem_id for Solution 1
        content = f"ID: {prod_id}. OEM ID: {oem_id}. {name}. {description}. Specs: {specs_str}"

        texts.append(content)
        text_ids.append(f"text-{prod_id}")
        text_metas.append({
            "id": prod_id,
            "oem_id": oem_id,
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
    async def process_image_batch(image_batch):
        images = []
        img_ids = []
        img_metas = []
        img_docs = []

        async with aiohttp.ClientSession() as session:
            for prod_id, oem_id, name, description, image_url, specs_str, idx in image_batch:
                try:
                    img_data = await fetch_image(session, image_url)
                    if img_data:
                        img = Image.open(BytesIO(img_data)).convert("RGB")
                        images.append(img)
                        img_ids.append(f"image-{prod_id}-{idx}")
                        img_docs.append(f"{name} (image)")
                        img_metas.append({
                            "id": prod_id,
                            "oem_id": oem_id,
                            "type": "image",
                            "name": name,
                            "description": description or "",
                            "images": image_url,
                            "specifications": specs_str
                        })
                except Exception as e:
                    print(f"⚠ Image failed for {prod_id} url={image_url}: {e}")

        if images:
            img_embs = clip_model.encode(images, normalize_embeddings=True, batch_size=IMAGE_BATCH_SIZE).tolist()
            image_collection.add(
                ids=img_ids,
                documents=img_docs,
                embeddings=img_embs,
                metadatas=img_metas
            )
            print(f"   🖼 Embedded {len(images)} images")

    # Process images in batches
    image_batch = []
    for prod_id, oem_id, name, description, images_field, specifications in rows:
        prod_id = str(prod_id)
        oem_id = str(oem_id) if oem_id is not None else ""
        specs_str = specs_to_string(specifications)
        image_list = normalize_image_list(images_field)
        images_str = ",".join(image_list)

        for idx, img_url in enumerate(image_list):
            image_batch.append((prod_id, oem_id, name, description, img_url, specs_str, idx))
            if len(image_batch) >= IMAGE_BATCH_SIZE:
                asyncio.run(process_image_batch(image_batch))
                image_batch = []
            time.sleep(SLEEP_DELAY)  # Minimal delay for server courtesy

    if image_batch:  # Process remaining images
        asyncio.run(process_image_batch(image_batch))
        image_batch = []

    offset += BATCH_SIZE
    if offset >= MAX_PRODUCTS:
        break

print(f"\n🎉 Done. Inserted ~{total_inserted} text products into ChromaDB.")
cur.close()
conn.close()