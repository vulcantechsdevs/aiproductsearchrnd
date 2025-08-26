from sentence_transformers import SentenceTransformer
import chromadb

# Load the same embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Chroma (must use the SAME persist_directory as insert script)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection
collection = chroma_client.get_or_create_collection(name="products")

# Function to search products
def search_products(query, top_k=1):
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print("\n🔎 Search Results for:", query)
    for i in range(len(results['ids'][0])):
        print(f"\nResult {i+1}:")
        print("ID:", results['ids'][0][i])
        print("Document:", results['documents'][0][i])
        print("Metadata:", results['metadatas'][0][i])
        print("Distance:", results['distances'][0][i])

# Example query
search_products("wireless headphones with noise cancellation")
