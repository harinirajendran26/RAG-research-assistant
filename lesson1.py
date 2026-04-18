import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Step 1: Your documents (imagine these are chunks from a PDF) ──────────────
documents = [
    "Virat Kohli is an Indian cricketer who plays for Royal Challengers Bangalore.",
    "The IPL was founded in 2008 by the BCCI.",
    "MS Dhoni is known for his calm captaincy and finishing abilities.",
    "Rohit Sharma holds the record for most centuries in ODI cricket.",
    "The Cricket World Cup is held every four years.",
]

print("Step 1: Embedding your documents...")
print("(First run downloads a small model ~90MB — one time only)\n")

# ── Step 2: Embed documents and store in ChromaDB (runs locally) ──────────────
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_texts(documents, embeddings_model)
print("Documents embedded and stored in ChromaDB!\n")

# ── Step 3: Query — RAG retrieves the relevant chunks ─────────────────────────
query = "Who is known for staying calm under pressure?"
results = vectorstore.similarity_search(query, k=2)  # top 2 matches

print(f"Query: {query}")
print("\nTop matching chunks from vector search:")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content}")

# ── Step 4: Pass retrieved chunks to Groq LLM to generate an answer ───────────
context = "\n".join([doc.page_content for doc in results])

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful assistant. Answer the user's question using ONLY the context provided. If the answer isn't in the context, say 'I don't know'."),
    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
]

print("\nSending to Groq LLM (Llama 3)...")
response = llm.invoke(messages)

print("\n── FINAL ANSWER ──────────────────────────────────────")
print(response.content)
print("──────────────────────────────────────────────────────")