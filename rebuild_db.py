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

# ── Step 1: Delete old ChromaDB ───────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print("Old ChromaDB deleted.")

# ── Step 2: Load PDF and merge all pages ─────────────────────────────────────
loader = PyPDFLoader("Attention.pdf")
pages = loader.load()
print(f"Pages loaded: {len(pages)}")

# Merge all pages into one string — NO cleaning yet
full_text = "\n\n".join([p.page_content for p in pages])

# ── Step 3: Find sections on RAW text (before any cleaning) ──────────────────
abstract_start    = full_text.find("Abstract\n")
intro_start       = full_text.find("1 Introduction\n")
background_start  = full_text.find("2 Background\n")
model_arch_start  = full_text.find("3 Model Architecture\n")

print(f"\nSection positions found in raw text:")
print(f"  Abstract        : {abstract_start}")
print(f"  1 Introduction  : {intro_start}")
print(f"  2 Background    : {background_start}")
print(f"  3 Model Arch    : {model_arch_start}")

print(f"\nRaw text sample around position 0-500:")
print(full_text[:500])
print(f"\nRaw text sample around position 500-1000:")
print(full_text[500:1000])

# ── Step 4: Extract sections cleanly ─────────────────────────────────────────
special_chunks = []

# Abstract — from "Abstract\n" to "1 Introduction\n"
if abstract_start != -1 and intro_start != -1:
    abstract_text = full_text[abstract_start:intro_start].strip()
    # Remove the footnotes (lines starting with ∗ or †) from abstract
    clean_abstract_lines = []
    for line in abstract_text.split('\n'):
        if line.strip().startswith('∗') or line.strip().startswith('†') or line.strip().startswith('‡'):
            continue
        clean_abstract_lines.append(line)
    abstract_text = '\n'.join(clean_abstract_lines).strip()

    special_chunks.append(Document(
        page_content=abstract_text,
        metadata={"section": "abstract", "page": 0}
    ))
    print(f"\nAbstract extracted ({len(abstract_text)} chars):")
    print(abstract_text[:400])
else:
    print("WARNING: Abstract not found!")

# Introduction — from "1 Introduction\n" to "2 Background\n"
if intro_start != -1 and background_start != -1:
    intro_text = full_text[intro_start:background_start].strip()
    special_chunks.append(Document(
        page_content=intro_text,
        metadata={"section": "introduction", "page": 1}
    ))
    print(f"\nIntroduction extracted ({len(intro_text)} chars)")
    print(intro_text[:200])
else:
    print("WARNING: Introduction not found!")

# ── Step 5: Remaining text — everything from Background onwards ───────────────
# This avoids duplicating abstract and intro in the chunked part
if background_start != -1:
    remaining_text = full_text[background_start:].strip()
elif model_arch_start != -1:
    remaining_text = full_text[model_arch_start:].strip()
else:
    remaining_text = full_text
    print("WARNING: Could not find background — chunking full text")

print(f"\nRemaining text to chunk: {len(remaining_text)} chars")

# ── Step 6: Clean only the remaining text ────────────────────────────────────
def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)        # remove emails
    text = re.sub(r'http\S+', '', text)         # remove URLs
    # Remove lines that are pure noise — very short AND not section headers
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep if longer than 15 chars
        if len(stripped) > 15:
            clean_lines.append(stripped)
        # Also keep section headers even if short (start with a digit)
        elif stripped and stripped[0].isdigit():
            clean_lines.append(stripped)
    return '\n'.join(clean_lines)

remaining_clean = clean_text(remaining_text)

# ── Step 7: Chunk only the remaining text ────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

remaining_doc = Document(
    page_content=remaining_clean,
    metadata={"section": "main", "page": 2}
)

regular_chunks = splitter.split_documents([remaining_doc])
print(f"Regular chunks from remaining text: {len(regular_chunks)}")

# ── Step 8: Combine — special chunks first ───────────────────────────────────
all_chunks = special_chunks + regular_chunks

print(f"\nTotal chunks in database : {len(all_chunks)}")
print(f"  Special (abstract+intro): {len(special_chunks)}")
print(f"  Regular (rest of paper) : {len(regular_chunks)}")

# ── Step 9: Preview chunks ───────────────────────────────────────────────────
print("\n── Chunk preview ──────────────────────────────────────")
for i, chunk in enumerate(all_chunks[:4]):
    print(f"\nChunk {i+1} | section: {chunk.metadata.get('section')} | chars: {len(chunk.page_content)}")
    print(chunk.page_content[:300])
    print("...")

# ── Step 10: Embed and save ───────────────────────────────────────────────────
print("\nEmbedding all chunks...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings_model,
    persist_directory=CHROMA_PATH
)

print(f"\nChromaDB rebuilt with {len(all_chunks)} clean chunks!")
print("Now run: python lesson3.py")