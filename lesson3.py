import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Load existing ChromaDB (no re-embedding needed) ───────────────────────────
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

print("ChromaDB loaded from disk!\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: MMR Search (Maximum Marginal Relevance)
# ─────────────────────────────────────────────────────────────────────────────
def search_mmr(question, k=3):
    """
    MMR balances relevance + diversity.
    fetch_k = how many candidates to consider
    k = how many final results to return
    lambda_mult = 0.5 means equal weight on relevance and diversity
                  closer to 1.0 = more relevance, closer to 0.0 = more diversity
    """
    results = vectorstore.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=10,
        lambda_mult=0.5
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Query Rewriting
# ─────────────────────────────────────────────────────────────────────────────
def rewrite_query(original_question):
    """
    Ask the LLM to rewrite a vague question into a
    specific search query that will match document content better.
    """
    messages = [
        SystemMessage(content="""You are a search query optimizer. 
        Rewrite the user's question into a short, specific search query 
        that will retrieve the most relevant content from a research paper.
        Return ONLY the rewritten query — no explanation, no punctuation."""),
        HumanMessage(content=f"Original question: {original_question}")
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Metadata Filtering
# ─────────────────────────────────────────────────────────────────────────────
def search_with_page_filter(question, page_range):
    """
    Only search chunks from specific pages.
    page_range = list of page numbers (0-indexed internally)
    Example: page_range=[0,1,2] searches pages 1-3
    """
    results = vectorstore.similarity_search(
        question,
        k=3,
        filter={"page": {"$in": page_range}}
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Generate answer from retrieved chunks
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(question, chunks):
    context = "\n\n".join([doc.page_content for doc in chunks])

    messages = [
        SystemMessage(content="""You are a helpful research assistant.
        Answer using ONLY the context provided.
        Cite the page number when you use information from a specific part.
        If the answer is not in the context, say 'I could not find this in the document.'"""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    response = llm.invoke(messages)
    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# Compare all 3 approaches side by side
# ─────────────────────────────────────────────────────────────────────────────
def compare_approaches(question):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # ── Approach 1: Basic search ──────────────────────────────────
    print("\n[1] BASIC SEARCH (Day 2 method)")
    basic_results = vectorstore.similarity_search(question, k=3)
    for i, doc in enumerate(basic_results, 1):
        section = doc.metadata.get('section', '?')
        page = doc.metadata.get('page', '?')
        print(f"  Chunk {i} → section:{section} page:{page}: {doc.page_content[:80]}...")

    # ── Approach 2: MMR search ────────────────────────────────────
    print("\n[2] MMR SEARCH (diverse results)")
    mmr_results = search_mmr(question, k=3)
    for i, doc in enumerate(mmr_results, 1):
        section = doc.metadata.get('section', '?')
        page = doc.metadata.get('page', '?')
        print(f"  Chunk {i} → section:{section} page:{page}: {doc.page_content[:80]}...")

    # ── Approach 3: Query rewriting + MMR ────────────────────────
    print("\n[3] QUERY REWRITING + MMR")
    rewritten = rewrite_query(question)
    print(f"  Original : {question}")
    print(f"  Rewritten: {rewritten}")
    smart_results = search_mmr(rewritten, k=3)
    for i, doc in enumerate(smart_results, 1):
        section = doc.metadata.get('section', '?')
        page = doc.metadata.get('page', '?')
        print(f"  Chunk {i} → section:{section} page:{page}: {doc.page_content[:80]}...")

    # ── Approach 4: Section-aware search (NEW) ───────────────────
    # For "main idea" type questions — directly target abstract
    print("\n[4] SECTION-AWARE SEARCH (smart routing)")
    main_idea_keywords = ["main idea", "summary", "overview", "about", "purpose", "what is this"]
    is_summary_question = any(kw in question.lower() for kw in main_idea_keywords)

    if is_summary_question:
        print("  Detected summary question → forcing abstract section")
        section_results = vectorstore.similarity_search(
            question,
            k=1,
            filter={"section": {"$eq": "abstract"}}
        )
        # Also get intro
        intro_results = vectorstore.similarity_search(
            question,
            k=1,
            filter={"section": {"$eq": "introduction"}}
        )
        final_results = section_results + intro_results
        for i, doc in enumerate(final_results, 1):
            section = doc.metadata.get('section', '?')
            print(f"  Chunk {i} → section:{section}: {doc.page_content[:80]}...")
    else:
        final_results = smart_results
        print("  Using rewriting + MMR results")

    # ── Final answer ──────────────────────────────────────────────
    print("\n── FINAL ANSWER ──────────────────────────────────────────")
    answer = generate_answer(question, final_results)
    print(answer)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Test the 3 approaches
# ─────────────────────────────────────────────────────────────────────────────

# Test 1 — the vague question that failed on Day 2
compare_approaches("What is the main idea of this document?")

# Test 2 — metadata filtering for early pages only
print(f"\n{'='*60}")
print("METADATA FILTER TEST — searching only pages 1-3")
print(f"{'='*60}")
filtered_results = search_with_page_filter(
    "What problem does this paper solve?",
    page_range=[0, 1, 2]   # pages 1-3 (0-indexed)
)
for i, doc in enumerate(filtered_results, 1):
    print(f"  Chunk {i} → page {doc.metadata.get('page', '?')+1}: {doc.page_content[:120]}...")

answer = generate_answer("What problem does this paper solve?", filtered_results)
print(f"\n── ANSWER ────────────────────────────────────────────")
print(answer)

# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop with smart retrieval
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Smart Q&A mode. Type 'quit' to exit. ──\n")
while True:
    user_input = input("Your question: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    if not user_input:
        continue

    # Route summary questions to abstract directly
    main_idea_keywords = ["main idea", "summary", "overview", "about",
                          "purpose", "what is this", "what is the paper"]
    is_summary = any(kw in user_input.lower() for kw in main_idea_keywords)

    if is_summary:
        print("  (Summary question detected → searching abstract + introduction)")
        results = vectorstore.similarity_search(
            user_input, k=1, filter={"section": {"$eq": "abstract"}}
        )
        results += vectorstore.similarity_search(
            user_input, k=1, filter={"section": {"$eq": "introduction"}}
        )
    else:
        rewritten = rewrite_query(user_input)
        print(f"  (Searching for: '{rewritten}')")
        results = search_mmr(rewritten, k=3)

    answer = generate_answer(user_input, results)
    print(f"\n── ANSWER ────────────────────────────────────────────")
    print(answer)
    print()