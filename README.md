# Multi-Source AI Research Assistant

A production-grade RAG application built over 10 days from scratch.
Upload PDFs, add web URLs, paste notes — ask questions across all sources
with cited answers, conversation memory, agentic tool use, and live RAGAS evaluation.

## Live Demo
https://rag-research-assistant-hzhpncsup3zwwdshykghfm.streamlit.app/

## Features
- Multi-source ingestion — PDFs, web URLs, plain text notes
- Cross-encoder re-ranking for precise retrieval
- Agentic mode — agent decides which tool to use per question
- Web search for live internet results
- RAGAS evaluation dashboard with 4 metrics
- Conversation memory for follow-up questions
- Source citations on every answer

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API (free) |
| Embeddings | BAAI/bge-base-en-v1.5 (768 dims, local) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | ChromaDB (local, versioned) |
| Agent | ReAct loop with tool selection |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |
| UI | Streamlit (3 tabs) |

## RAG Techniques Used
- Section-aware chunking — abstract and introduction as dedicated chunks
- BGE embeddings with 768 dimensions for strong semantic matching
- Cross-encoder re-ranking — two-stage retrieval pipeline
- MMR search — balances relevance and diversity
- Question-type routing — summary vs concept vs live queries
- Conversation memory — last 4 exchanges passed as context
- Agentic tool selection — PDF search, web search, calculator
- RAGAS evaluation — real metrics not just eyeballing

## Project Structure
├── app.py                # Streamlit UI — Chat, Evaluation, About tabs
├── rag_engine.py         # Core RAG: multi-source ingestion, retrieval, generation
├── agent.py              # Agentic layer: tool selection and chaining
├── evaluator.py          # RAGAS evaluation pipeline
├── lesson1.py            # Day 1: first RAG pipeline (5 strings → answer)
├── lesson2.py            # Day 2: real PDF ingestion with chunking
├── lesson3.py            # Day 3: MMR, query rewriting, section routing
├── lesson6_rerank.py     # Day 6: cross-encoder re-ranking
├── lesson6_ragas.py      # Day 6: RAGAS evaluation
├── lesson7_agent.py      # Day 7: agentic RAG with tool use
├── rebuild_db.py         # Utility: rebuild ChromaDB from scratch
├── reembed.py            # Utility: upgrade embedding model
├── debug_db.py           # Utility: inspect ChromaDB contents
├── requirements.txt      # Python dependencies
└── README.md             # This file

## Setup

```bash
git clone https://github.com/harinirajendran26/rag-research-assistant.git
cd rag-research-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
GROQ_API_KEY=your_groq_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

Get free keys at [console.groq.com](https://console.groq.com) and [huggingface.co](https://huggingface.co)

Run the app:
```bash
streamlit run app.py
```

## Evaluation Results

Evaluated on the Attention Is All You Need paper:

| Metric | Score | Meaning |
|---|---|---|
| Faithfulness | 0.93 | LLM stays grounded — minimal hallucination |
| Answer Relevancy | 0.74 | Answers address the questions asked |
| Context Precision | 0.85 | Retrieved chunks are relevant |
| Context Recall | 1.00 | All necessary information was found |

## Learning Journey

| Day | What was built |
|---|---|
| Day 1 | First RAG pipeline — embeddings, ChromaDB, Groq |
| Day 2 | Real PDF ingestion with chunking and persistence |
| Day 3 | Smart retrieval — MMR, query rewriting, section routing |
| Day 4 | Multi-PDF Streamlit UI |
| Day 5 | Conversation memory, polish, first deployment |
| Day 6 | Cross-encoder re-ranking + RAGAS evaluation |
| Day 7 | Agentic RAG with tool selection and web search |
| Days 8-10 | Capstone — all techniques combined |

## Key Lessons Learned
- Chunking quality beats retrieval algorithm quality every time
- Never clean text before extracting document structure
- Embedding model dimensions must match at build and query time
- Production RAG routes different question types differently
- Always evaluate with real metrics, not just eyeballing answers