import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

CHROMA_PATH = "./chroma_db"
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print("Old ChromaDB deleted.")

# ── Load and merge PDF ────────────────────────────────────────────────────────
loader = PyPDFLoader("Attention.pdf")
pages = loader.load()
full_text = "\n\n".join([p.page_content for p in pages])
print(f"Pages loaded: {len(pages)}")

# ── Extract sections on raw text ──────────────────────────────────────────────
abstract_start   = full_text.find("Abstract\n")
intro_start      = full_text.find("1 Introduction\n")
background_start = full_text.find("2 Background\n")

special_chunks = []

# Abstract
if abstract_start != -1 and intro_start != -1:
    abstract_text = full_text[abstract_start:intro_start].strip()
    # Remove footnote lines
    lines = [l for l in abstract_text.split('\n')
             if not l.strip().startswith(('∗','†','‡','31st','arXiv'))]
    abstract_text = '\n'.join(lines).strip()
    special_chunks.append(Document(
        page_content=abstract_text,
        metadata={"section": "abstract", "page": 0}
    ))
    print(f"Abstract extracted ({len(abstract_text)} chars)")

# Introduction
if intro_start != -1 and background_start != -1:
    intro_text = full_text[intro_start:background_start].strip()
    special_chunks.append(Document(
        page_content=intro_text,
        metadata={"section": "introduction", "page": 1}
    ))
    print(f"Introduction extracted ({len(intro_text)} chars)")

# ── Remaining text ────────────────────────────────────────────────────────────
remaining_text = full_text[background_start:].strip()

def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 15:
            clean_lines.append(stripped)
        elif stripped and stripped[0].isdigit():
            clean_lines.append(stripped)
    return '\n'.join(clean_lines)

remaining_clean = clean_text(remaining_text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

regular_chunks = splitter.split_documents([
    Document(page_content=remaining_clean, metadata={"section": "main", "page": 2})
])

all_chunks = special_chunks + regular_chunks
print(f"Total chunks: {len(all_chunks)}")

# ── UPGRADED embedding model ──────────────────────────────────────────────────
print("\nLoading upgraded embedding model...")
print("(Downloads ~500MB first time — much more powerful than MiniLM)\n")

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"  # much stronger than all-MiniLM-L6-v2
)

vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings_model,
    persist_directory=CHROMA_PATH
)

print(f"\nDone! ChromaDB rebuilt with {len(all_chunks)} chunks using BGE embeddings.")
print("Now update lesson3.py embedding model name and run it.")