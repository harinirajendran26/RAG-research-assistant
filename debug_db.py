from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_BASE_PATH = "./chroma_db_multi"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"

# ── Find latest versioned DB ──────────────────────────────────────────────────
i = 1
latest_path = None
while True:
    path = f"{CHROMA_BASE_PATH}_v{i}"
    if os.path.exists(path):
        latest_path = path
        i += 1
    else:
        break

if not latest_path:
    print("No ChromaDB found. Run the app and process a PDF first.")
    exit()

print(f"Found DB at: {latest_path}\n")

# ── Load with correct model ───────────────────────────────────────────────────
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=latest_path,
    embedding_function=embeddings_model
)

# ── Check 1: Total chunks ─────────────────────────────────────────────────────
collection = vectorstore._collection
count = collection.count()
print(f"Total chunks in ChromaDB: {count}")

# ── Check 2: First 5 chunks ───────────────────────────────────────────────────
print("\nFirst 5 chunks:")
results = collection.get(limit=5)
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Metadata : {meta}")
    print(f"Content  : {doc[:200]}")

# ── Check 3: Similarity search ────────────────────────────────────────────────
print("\n\nSimilarity search — 'attention mechanism':")
raw = vectorstore.similarity_search("attention mechanism", k=3)
for i, doc in enumerate(raw, 1):
    print(f"\nResult {i}:")
    print(f"  Metadata : {doc.metadata}")
    print(f"  Content  : {doc.page_content[:150]}")

# ── Check 4: Abstract filter ──────────────────────────────────────────────────
print("\n\nAbstract section filter:")
abstract = vectorstore.similarity_search(
    "main idea", k=1,
    filter={"section": {"$eq": "abstract"}}
)
for doc in abstract:
    print(f"  Section : {doc.metadata.get('section')}")
    print(f"  Source  : {doc.metadata.get('source')}")
    print(f"  Content : {doc.page_content[:300]}")

# ── Check 5: All unique sources ───────────────────────────────────────────────
print("\n\nAll unique sources in DB:")
all_results = collection.get()
sources = set()
for meta in all_results['metadatas']:
    if meta and 'source' in meta:
        sources.add(meta['source'])
for src in sorted(sources):
    print(f"  📄 {src}")