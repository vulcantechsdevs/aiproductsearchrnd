import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
import json
import requests
from PIL import Image
from io import BytesIO

# Connect to Postgres
conn = psycopg2.connect(
    dbname="medworld",
    user="postgres",
    password="1",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Fetch product data (limit for testing)
cur.execute("SELECT id, name, description, images, specifications FROM products.products_info LIMIT 100;")
rows = cur.fetchall()

# Load CLIP model (can handle text + image)
clip_model = SentenceTransformer("clip-ViT-B-32")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collections
for coll in ["products_text", "products_image"]:
    try:
        chroma_client.delete_collection(coll)
        print(f"🗑️ Old collection {coll} deleted")
    except:
        pass

# Create new collections
text_collection = chroma_client.create_collection(name="products_text")
image_collection = chroma_client.create_collection(name="products_image")

for row in rows:
    prod_id, name, description, images, specifications = row
    content = f"{name}: {description}"

    # ✅ Text embedding
    text_embedding = clip_model.encode(content).tolist()

    # ✅ Handle images (download & embed)
    image_embeddings = []
    image_str_list = []

    if images:
        if isinstance(images, list):
            image_list = images
        else:
            image_list = [str(images)]

        for img_url in image_list:
            try:
                response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                emb = clip_model.encode(img).tolist()
                image_embeddings.append((img_url, emb))
                image_str_list.append(img_url)
            except Exception as e:
                print(f"⚠️ Failed to process image {img_url} for product {prod_id}: {e}")

    # ✅ Handle specifications
    specs_str = "[]"
    if specifications:
        try:
            specs_str = json.dumps(specifications if isinstance(specifications, list) else specifications)
        except Exception as e:
            print(f"⚠️ Could not serialize specifications for product {prod_id}: {e}")

    # ✅ Add text embedding to text collection
    text_collection.add(
        ids=[f"text-{prod_id}"],
        documents=[content],
        embeddings=[text_embedding],
        metadatas=[{
            "id": prod_id,
            "type": "text",
            "name": name,
            "description": description,
            "images": ",".join(image_str_list),
            "specifications": specs_str
        }]
    )

    # ✅ Add image embeddings to image collection - FIXED: Use consistent metadata structure
    for idx, (img_url, emb) in enumerate(image_embeddings):
        image_collection.add(
            ids=[f"image-{prod_id}-{idx}"],
            documents=[f"{name} (image)"],
            embeddings=[emb],
            metadatas=[{
                "id": prod_id,
                "type": "image",
                "name": name,
                "description": description,
                "images": ",".join(image_str_list),  # CHANGED: Use "images" (plural) with all images
                "specifications": specs_str
            }]
        )

    print(f"✅ Added product: {name}")

print("🎉 Text + Image Data inserted into Chroma successfully!")

cur.close()
conn.close()