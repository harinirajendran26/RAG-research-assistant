import streamlit as st
import tempfile
import os
from rag_engine import (
    load_embeddings_model, load_reranker, load_llm,
    load_vectorstore, process_sources,
    answer, get_loaded_sources
)
from agent import run_agent
from evaluator import run_evaluation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Source Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.source-pill {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:0.75rem; margin:2px; font-weight:500;
}
.pdf-pill   { background:#E1F5EE; color:#085041; }
.web-pill   { background:#EEEDFE; color:#3C3489; }
.text-pill  { background:#FAEEDA; color:#633806; }
.metric-box {
    background:var(--color-background-secondary);
    border-radius:8px; padding:12px; text-align:center;
}
.metric-num { font-size:1.8rem; font-weight:600; }
.metric-lbl { font-size:0.75rem; color:#888; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── Cache models ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings(): return load_embeddings_model()

@st.cache_resource
def get_reranker():   return load_reranker()

@st.cache_resource
def get_llm():        return load_llm()

embeddings = get_embeddings()
reranker   = get_reranker()
llm        = get_llm()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "vectorstore": load_vectorstore(embeddings),
    "messages":    [],
    "doc_stats":   {},
    "eval_results": None,
    "use_agent":   False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown("*Multi-source RAG + Agent + Evaluation*")
    st.markdown("---")

    # Agent toggle
    st.session_state.use_agent = st.toggle(
        "🤖 Agentic mode",
        value=st.session_state.use_agent,
        help="Agent decides which tools to use. Slower but smarter."
    )

    st.markdown("---")
    st.markdown("### Add Sources")

    # PDF upload
    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    # URL input
    url_input = st.text_area(
        "Add web URLs (one per line)",
        placeholder="https://example.com/article\nhttps://...",
        height=80
    )

    # Text input
    text_input = st.text_area(
        "Paste notes or text",
        placeholder="Paste any text content here...",
        height=80
    )

    if st.button("⚙️ Process Sources", type="primary",
                 use_container_width=True):
        pdf_paths   = []
        urls        = [u.strip() for u in url_input.split('\n')
                       if u.strip().startswith('http')]
        text_inputs = [text_input.strip()] if text_input.strip() else []

        # Save PDFs to temp files
        for uf in (uploaded or []):
            tmp = os.path.join(tempfile.gettempdir(), uf.name)
            with open(tmp, "wb") as f:
                f.write(uf.read())
            pdf_paths.append(tmp)

        if not pdf_paths and not urls and not text_inputs:
            st.warning("Add at least one source.")
        else:
            with st.spinner("Processing sources..."):
                summary, vs = process_sources(
                    pdf_paths, urls, text_inputs, embeddings
                )
                st.session_state.vectorstore = vs
                st.session_state.doc_stats   = summary
                st.session_state.messages    = []
                st.session_state.eval_results = None

            # Cleanup temp files
            for p in pdf_paths:
                try: os.remove(p)
                except: pass

            st.success(f"✅ Processed {len(summary)} source(s)!")
            for name, info in summary.items():
                typ = info.get('type','?')
                st.markdown(
                    f"<span class='source-pill {typ}-pill'>{typ}</span> "
                    f"**{name[:40]}** — {info.get('chunks',0)} chunks",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # Loaded sources
    loaded = get_loaded_sources(st.session_state.vectorstore)
    if loaded:
        st.markdown("### Loaded Sources")
        for src in loaded:
            typ = src['type']
            st.markdown(
                f"<span class='source-pill {typ}-pill'>{typ}</span> "
                f"{src['name'][:40]}",
                unsafe_allow_html=True
            )

    if st.session_state.messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("LangChain · ChromaDB · BGE · Groq · RAGAS")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Evaluation", "ℹ️ About"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask anything about your sources")

    mode = "🤖 Agent mode" if st.session_state.use_agent else "⚡ Fast mode"
    if loaded:
        st.caption(
            f"{mode} · {len(loaded)} source(s) loaded · "
            f"{len(st.session_state.messages)//2} questions asked"
        )
    else:
        st.info("👈 Add sources in the sidebar to get started.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # Show agent steps if available
                if msg.get("steps"):
                    with st.expander("🔍 Agent reasoning steps"):
                        for i, step in enumerate(msg["steps"], 1):
                            st.markdown(
                                f"**Step {i}** — `{step['tool']}`  \n"
                                f"Input: {step['input'][:80]}  \n"
                                f"Result: {step['observation'][:120]}"
                            )
                # Show sources
                if msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for s in msg["sources"]:
                            typ = s.get('type','?')
                            st.markdown(
                                f"<span class='source-pill {typ}-pill'>"
                                f"{typ}</span> **{s['source']}** "
                                f"— *{s['section']}*  \n"
                                f"<small>{s['preview']}</small>",
                                unsafe_allow_html=True
                            )

    # Chat input
    if prompt := st.chat_input(
        "Ask anything...",
        disabled=not loaded
    ):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                "Agent thinking..." if st.session_state.use_agent
                else "Searching..."
            ):
                if st.session_state.use_agent:
                    result = run_agent(
                        prompt,
                        st.session_state.messages[:-1],
                        st.session_state.vectorstore,
                        reranker, llm
                    )
                else:
                    result = answer(
                        prompt,
                        st.session_state.messages[:-1],
                        st.session_state.vectorstore,
                        reranker, llm
                    )

            st.markdown(result["answer"])

            if result.get("steps"):
                with st.expander("🔍 Agent reasoning steps"):
                    for i, step in enumerate(result["steps"], 1):
                        st.markdown(
                            f"**Step {i}** — `{step['tool']}`  \n"
                            f"Input: {step['input'][:80]}  \n"
                            f"Result: {step['observation'][:120]}"
                        )

            if result.get("sources"):
                with st.expander("📎 Sources"):
                    for s in result["sources"]:
                        typ = s.get('type', '?')
                        st.markdown(
                            f"<span class='source-pill {typ}-pill'>"
                            f"{typ}</span> **{s['source']}** "
                            f"— *{s['section']}*  \n"
                            f"<small>{s['preview']}</small>",
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "steps":   result.get("steps", [])
        })

# ── Tab 2: Evaluation ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### RAGAS Evaluation Dashboard")
    st.markdown(
        "Run evaluation to measure your RAG system's quality "
        "with real metrics."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        eval_questions = st.text_area(
            "Test questions (one per line)",
            value=(
                "What is the main idea of this document?\n"
                "What methods or techniques are described?\n"
                "What were the key results or findings?"
            ),
            height=120
        )
        eval_truths = st.text_area(
            "Ground truths (one per line, matching order)",
            value=(
                "The document proposes a new approach based on attention mechanisms.\n"
                "The paper describes transformer architecture with multi-head attention.\n"
                "The model achieved state of the art results on translation benchmarks."
            ),
            height=120
        )

    with col2:
        st.markdown("**What each metric means:**")
        st.markdown("🟢 **Faithfulness** — no hallucination")
        st.markdown("🟡 **Answer Relevancy** — on-topic answers")
        st.markdown("🔵 **Context Precision** — clean retrieval")
        st.markdown("🟣 **Context Recall** — found everything")
        st.markdown("")
        st.markdown("*Good scores: > 0.7*")

    if st.button("▶ Run Evaluation", type="primary",
                 disabled=not loaded):
        questions = [q.strip() for q in eval_questions.split('\n')
                     if q.strip()]
        truths    = [t.strip() for t in eval_truths.split('\n')
                     if t.strip()]

        if len(questions) != len(truths):
            st.error("Number of questions must match number of ground truths.")
        else:
            with st.spinner(
                f"Evaluating {len(questions)} questions — "
                "takes 2-4 minutes..."
            ):
                # Generate answers for evaluation
                answers_list  = []
                contexts_list = []

                for q in questions:
                    res = answer(
                        q, [],
                        st.session_state.vectorstore,
                        reranker, llm
                    )
                    answers_list.append(res["answer"])
                    contexts_list.append([
                        c.page_content for c in res["chunks"]
                    ])

                eval_results = run_evaluation(
                    questions, truths,
                    answers_list, contexts_list,
                    llm, embeddings
                )
                st.session_state.eval_results = eval_results

    # Show results
    if st.session_state.eval_results:
        res = st.session_state.eval_results
        if res.get("error"):
            st.error(f"Evaluation error: {res['error']}")
        else:
            st.markdown("---")
            st.markdown("#### Results")
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, "Faithfulness",      res["faithfulness"],      "🟢"),
                (c2, "Answer Relevancy",  res["answer_relevancy"],  "🟡"),
                (c3, "Context Precision", res["context_precision"], "🔵"),
                (c4, "Context Recall",    res["context_recall"],    "🟣"),
            ]
            for col, name, score, icon in metrics:
                with col:
                    if score is not None:
                        color = "#1D9E75" if score > 0.7 \
                                else "#EF9F27" if score > 0.5 \
                                else "#D85A30"
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-num' "
                            f"style='color:{color}'>"
                            f"{icon} {score:.2f}</div>"
                            f"<div class='metric-lbl'>{name}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-num'>N/A</div>"
                            f"<div class='metric-lbl'>{name}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Multi-Source AI Research Assistant")
    st.markdown("""
**What this does:**
Upload PDFs, add web URLs, paste text notes — then ask questions
across all sources at once. Answers always cite which source
and section they came from.

**RAG techniques used:**
- Section-aware chunking (abstract + intro extracted separately)
- BGE embeddings (768-dim, local, strong semantic matching)
- Cross-encoder re-ranking (ms-marco-MiniLM)
- MMR retrieval (balances relevance + diversity)
- Query routing by question type
- Conversation memory (last 4 exchanges)
- Agentic tool use (PDF search + web search + calculator)
- RAGAS evaluation (faithfulness, relevancy, precision, recall)

**Tech stack:**
LangChain · ChromaDB · BAAI/bge-base-en-v1.5 ·
cross-encoder/ms-marco-MiniLM-L-6-v2 · Groq Llama 3.3 70B ·
RAGAS · Streamlit
    """)