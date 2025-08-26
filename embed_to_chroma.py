import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
import json

# Connect to Postgres
conn = psycopg2.connect(
    dbname="medworld",
    user="postgres",      # change if your user is different
    password="1",         # replace with your actual password
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Fetch product data (limit for testing)
cur.execute("SELECT id, name, description, images, specifications FROM products.products_info LIMIT 100;")
rows = cur.fetchall()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection & create new
try:
    chroma_client.delete_collection("products")
    print("🗑️ Old collection deleted")
except:
    pass

collection = chroma_client.create_collection(name="products")



for row in rows:
    # Adjust unpacking based on your SELECT query
    prod_id, name, description, images, specifications = row  
    content = f"{name}: {description}"
    embedding = model.encode(content).tolist()

    # ✅ Handle images: store as comma-separated string
    image_str = None
    if images:
        if isinstance(images, list):   # if already a list
            image_str = ",".join(images)
        else:                          # if it's a single string
            image_str = str(images)

    # ✅ Handle specifications: store as JSON string
    specs_str = None
    if specifications:
        try:
            # Ensure it's always valid JSON string
            specs_str = json.dumps(specifications if isinstance(specifications, list) else specifications)
        except Exception as e:
            print(f"⚠️ Could not serialize specifications for product {prod_id}: {e}")
            specs_str = "[]"

    # Add to Chroma
    collection.add(
        ids=[str(prod_id)],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "name": name,
            "description": description,
            "images": image_str,
            "specifications": specs_str
        }]
    )
    print(f"✅ Added product: {name}")

print("🎉 Data inserted into Chroma successfully!")

cur.close()
conn.close()
