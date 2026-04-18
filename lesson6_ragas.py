import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

CHROMA_BASE_PATH = "./chroma_db_multi"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
reranker         = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm              = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Load ChromaDB ─────────────────────────────────────────────────────────────
i = 1
latest_path = None
while True:
    path = f"{CHROMA_BASE_PATH}_v{i}"
    if os.path.exists(path):
        latest_path = path
        i += 1
    else:
        break

vectorstore = Chroma(
    persist_directory=latest_path,
    embedding_function=embeddings_model
)
print(f"Loaded DB: {latest_path}\n")


# ── Helper functions ──────────────────────────────────────────────────────────
def retrieve_with_reranking(query: str, fetch_k=20, top_k=4):
    candidates = vectorstore.similarity_search(query, k=fetch_k)
    pairs      = [(query, doc.page_content) for doc in candidates]
    scores     = reranker.predict(pairs)
    scored     = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join([doc.page_content for doc in chunks])
    messages = [
        SystemMessage(content="""Answer using ONLY the provided context.
If not in context say 'I could not find this in the documents.'"""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    return llm.invoke(messages).content


# ── Test questions — ground truth for evaluation ──────────────────────────────
# These are questions where we KNOW the correct answer
# RAGAS uses these to measure how well your system performs
test_questions = [
    "What is the Transformer architecture?",
    "How does multi-head attention work?",
    "What training data was used for the translation tasks?",
    "What BLEU score did the model achieve on English to German translation?",
    "What is the purpose of positional encoding?",
]

ground_truths = [
    "The Transformer is a model architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "Multi-head attention runs attention in parallel across multiple heads, allowing the model to attend to different representation subspaces.",
    "The WMT 2014 English-German and English-French datasets were used for training.",
    "The model achieved 28.4 BLEU on the WMT 2014 English-to-German translation task.",
    "Positional encoding adds information about the position of tokens in the sequence since the model has no recurrence or convolution.",
]

# ── Run evaluation pipeline ───────────────────────────────────────────────────
print("Running RAG pipeline on test questions...")
print("This makes API calls — takes 2-3 minutes\n")

questions_list  = []
answers_list    = []
contexts_list   = []
ground_truth_list = []

for question, truth in zip(test_questions, ground_truths):
    print(f"Processing: {question[:50]}...")

    # Retrieve with re-ranking
    chunks = retrieve_with_reranking(question)

    # Generate answer
    answer = generate_answer(question, chunks)

    # Collect contexts as list of strings (RAGAS format)
    contexts = [doc.page_content for doc in chunks]

    questions_list.append(question)
    answers_list.append(answer)
    contexts_list.append(contexts)
    ground_truth_list.append(truth)

# ── Build RAGAS dataset ───────────────────────────────────────────────────────
print("\nBuilding evaluation dataset...")
eval_dataset = Dataset.from_dict({
    "question":    questions_list,
    "answer":      answers_list,
    "contexts":    contexts_list,
    "ground_truth": ground_truth_list,
})

# ── Wrap models for RAGAS ─────────────────────────────────────────────────────
# RAGAS needs LLM and embeddings to compute its metrics
ragas_llm        = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)

# ── Run RAGAS evaluation ──────────────────────────────────────────────────────
print("Running RAGAS evaluation...")
print("(Makes additional LLM calls to score each metric)\n")

results = evaluate(
    dataset  = eval_dataset,
    metrics  = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ],
    llm         = ragas_llm,
    embeddings  = ragas_embeddings,
)
 

# ── Print results ─────────────────────────────────────────────────────────────

df = results.to_pandas()

print("\n" + "="*60)
print("RAGAS EVALUATION RESULTS")
print("="*60)

# Get scores safely — handle both old and new RAGAS formats
def get_score(df, col):
    if col in df.columns:
        val = df[col].mean()
        return float(val) if not hasattr(val, '__iter__') else float(val.iloc[0])
    return None

faithfulness_score      = get_score(df, 'faithfulness')
answer_relevancy_score  = get_score(df, 'answer_relevancy')
context_precision_score = get_score(df, 'context_precision')
context_recall_score    = get_score(df, 'context_recall')

# Print all column names so we can see what RAGAS actually returns
print(f"\nColumns in results: {list(df.columns)}")

print(f"\nAverage scores across {len(df)} questions:")
print(f"  Faithfulness      : {faithfulness_score:.3f}" if faithfulness_score else "  Faithfulness      : N/A")
print(f"  Answer Relevancy  : {answer_relevancy_score:.3f}" if answer_relevancy_score else "  Answer Relevancy  : N/A")
print(f"  Context Precision : {context_precision_score:.3f}" if context_precision_score else "  Context Precision : N/A")
print(f"  Context Recall    : {context_recall_score:.3f}" if context_recall_score else "  Context Recall    : N/A")

print("\n── What these scores mean ─────────────────────────────")
if faithfulness_score:
    print(f"Faithfulness {faithfulness_score:.2f}:", end=" ")
    if faithfulness_score > 0.8:   print("GOOD — LLM staying grounded in context")
    elif faithfulness_score > 0.6: print("OK — some hallucination present")
    else:                          print("POOR — LLM hallucinating significantly")

if answer_relevancy_score:
    print(f"Answer Relevancy {answer_relevancy_score:.2f}:", end=" ")
    if answer_relevancy_score > 0.8:   print("GOOD — answers address questions directly")
    elif answer_relevancy_score > 0.6: print("OK — answers somewhat relevant")
    else:                              print("POOR — answers are off topic")

if context_precision_score:
    print(f"Context Precision {context_precision_score:.2f}:", end=" ")
    if context_precision_score > 0.7:   print("GOOD — retrieved chunks are relevant")
    elif context_precision_score > 0.5: print("OK — some irrelevant chunks retrieved")
    else:                               print("POOR — retrieval pulling irrelevant content")

if context_recall_score:
    print(f"Context Recall {context_recall_score:.2f}:", end=" ")
    if context_recall_score > 0.7:   print("GOOD — finding all necessary information")
    elif context_recall_score > 0.5: print("OK — missing some relevant information")
    else:                            print("POOR — missing significant information")

print("\n── Per question breakdown ─────────────────────────────")
print(f"\n{'Q':<4} {'Question':<45} {'Faith':>6} {'Relev':>6}")
print("-" * 65)
for i, row in df.iterrows():
    q_short = questions_list[i][:43] + ".."
    faith   = f"{row['faithfulness']:.2f}"   if 'faithfulness'     in row and not hasattr(row['faithfulness'], '__iter__')     else "N/A"
    relev   = f"{row['answer_relevancy']:.2f}" if 'answer_relevancy' in row and not hasattr(row['answer_relevancy'], '__iter__') else "N/A"
    print(f"Q{i+1:<3} {q_short:<45} {faith:>6} {relev:>6}")

print("\n── Also fix the deprecated imports ─────────────────────")
print("Run: pip install --upgrade ragas")
print("Then update imports to use ragas.metrics.collections")
print("\nFull results DataFrame:")
print(df.to_string())