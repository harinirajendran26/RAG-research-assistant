import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_BASE_PATH = "./chroma_db_multi"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Loading re-ranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Load latest ChromaDB ──────────────────────────────────────────────────────
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
    print("No ChromaDB found. Run app.py and process a PDF first.")
    exit()

vectorstore = Chroma(
    persist_directory=latest_path,
    embedding_function=embeddings_model
)
print(f"Loaded ChromaDB from: {latest_path}\n")


# ── Re-ranking function ───────────────────────────────────────────────────────
def retrieve_and_rerank(query: str, fetch_k: int = 20, top_k: int = 4):
    """
    Two-stage retrieval:
    1. Fetch fetch_k candidates using fast embedding search
    2. Re-rank with cross-encoder, return top_k
    """
    # Stage 1 — fast embedding retrieval (cast wide net)
    print(f"\nStage 1: Fetching {fetch_k} candidates with BGE embeddings...")
    candidates = vectorstore.similarity_search(query, k=fetch_k)
    print(f"Got {len(candidates)} candidates")

    # Stage 2 — re-rank with cross-encoder
    print(f"Stage 2: Re-ranking with cross-encoder...")

    # Cross-encoder scores each (query, chunk) pair together
    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    # Sort by score descending
    scored_docs = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    # Return top_k after re-ranking
    top_docs = [doc for score, doc in scored_docs[:top_k]]
    top_scores = [score for score, doc in scored_docs[:top_k]]

    return top_docs, top_scores


# ── Generate answer ───────────────────────────────────────────────────────────
def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join([
        f"[From: {doc.metadata.get('source','unknown')}, "
        f"Section: {doc.metadata.get('section','?')}]\n{doc.page_content}"
        for doc in chunks
    ])

    messages = [
        SystemMessage(content="""You are a helpful research assistant.
Answer using ONLY the provided context.
Cite which document your answer comes from.
If not in context say 'I could not find this in the documents.'"""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    return llm.invoke(messages).content


# ── Compare: without vs with re-ranking ──────────────────────────────────────
def compare_retrieval(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Without re-ranking (basic MMR — what we had before)
    print("\n[WITHOUT RE-RANKING] Basic MMR top 4:")
    basic_chunks = vectorstore.max_marginal_relevance_search(
        question, k=4, fetch_k=20, lambda_mult=0.5
    )
    for i, doc in enumerate(basic_chunks, 1):
        print(f"  {i}. [{doc.metadata.get('section','?')}] "
              f"{doc.page_content[:80]}...")

    basic_answer = generate_answer(question, basic_chunks)
    print(f"\nAnswer: {basic_answer[:300]}...")

    # With re-ranking
    print(f"\n[WITH RE-RANKING] Cross-encoder top 4:")
    reranked_chunks, scores = retrieve_and_rerank(question, fetch_k=20, top_k=4)
    for i, (doc, score) in enumerate(zip(reranked_chunks, scores), 1):
        print(f"  {i}. score={score:.3f} [{doc.metadata.get('section','?')}] "
              f"{doc.page_content[:80]}...")

    reranked_answer = generate_answer(question, reranked_chunks)
    print(f"\nAnswer: {reranked_answer[:300]}...")

    return basic_chunks, reranked_chunks


# ── Run comparison ────────────────────────────────────────────────────────────
compare_retrieval("How does multi-head attention work?")
compare_retrieval("What are the training details and hyperparameters?")
compare_retrieval("What problem does this paper solve?")