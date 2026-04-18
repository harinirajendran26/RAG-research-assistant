import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Step 1: Load the PDF ──────────────────────────────────────────────────────
PDF_PATH = "Attention.pdf"
print(f"Loading PDF: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Pages loaded: {len(pages)}")
print(f"Sample text from page 1:\n{pages[0].page_content[:300]}\n")

# ── Step 2: Split into chunks ─────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # max characters per chunk
    chunk_overlap=200,     # overlap between consecutive chunks
    separators=["\n\n", "\n", " ", ""]  # tries to split on paragraphs first
)

chunks = splitter.split_documents(pages)
print(f"Total chunks created: {len(chunks)}")
print(f"Sample chunk:\n{chunks[0].page_content}\n")


# ── Step 3: Embed chunks OR load from disk if already done ───────────────────
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_PATH = "./chroma_db"

if os.path.exists(CHROMA_PATH):
    print("Loading existing ChromaDB from disk (no re-embedding needed)...\n")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )
else:
    print("Embedding chunks... (may take 30–60 seconds for large PDFs)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=CHROMA_PATH
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB\n")

# ── Step 4: Ask questions about your PDF ─────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def ask(question):
    # Retrieve top 3 relevant chunks
    results = vectorstore.similarity_search(question, k=3)

    # Show what was retrieved
    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks:")
    for i, doc in enumerate(results, 1):
        print(f"  Chunk {i} (page {doc.metadata.get('page', '?')+1}):")
        print(f"  {doc.page_content[:150]}...")

    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])

    # Generate answer
    messages = [
        SystemMessage(content="""You are a helpful assistant that answers questions 
        about documents. Answer using ONLY the provided context. 
        Always mention which part of the document supports your answer.
        If the answer is not in the context, say 'I could not find this in the document.'"""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)

    print(f"\n── ANSWER ────────────────────────────────────────────")
    print(response.content)
    print(f"──────────────────────────────────────────────────────\n")

# ── Step 5: Ask multiple questions ───────────────────────────────────────────
print("\n── Interactive mode. Type your questions. Type 'quit' to exit. ──\n")
while True:
    user_input = input("Your question: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    if not user_input:
        continue
    ask(user_input)