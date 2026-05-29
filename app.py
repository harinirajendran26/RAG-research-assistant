import streamlit as st
import tempfile
import os
import time
import datetime
from rag_engine import (
    load_embeddings_model, load_reranker, load_llm,
    load_vectorstore, process_sources,
    answer, answer_stream, get_loaded_sources
)
from agent import run_agent

try:
    from evaluator import run_evaluation
    EVAL_AVAILABLE = True
except Exception:
    EVAL_AVAILABLE = False
    def run_evaluation(*args, **kwargs):
        return {"error": "Evaluation unavailable", "faithfulness": None,
                "answer_relevancy": None, "context_precision": None,
                "context_recall": None, "dataframe": None}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 960px !important;
}
.stApp { background: #fdf6ee !important; }
.stApp p, .stApp span, .stApp div,
.stApp label, .stApp li { color: #3d0c11; }
.stMarkdown p  { color: #3d0c11 !important; }
.stMarkdown li { color: #3d0c11 !important; }

[data-testid="stSidebar"] {
    background: #3d0c11 !important;
    border-right: 1px solid #6b1a22;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #c8a882 !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #8a4a52 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] hr { border-color: #6b1a22 !important; opacity: 0.5; }
[data-testid="stSidebar"] [data-testid="stToggle"] {
    background: #4d1017;
    border-radius: 8px;
    padding: 4px 8px;
}
[data-testid="stSidebar"] .stButton > button {
    background: #8b1a1a !important;
    color: #fdf6ee !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #a02020 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 1.5px dashed #6b1a22;
    border-radius: 10px;
    background: #4d1017;
}
[data-testid="stSidebar"] textarea {
    background: #4d1017 !important;
    border: 1px solid #6b1a22 !important;
    border-radius: 8px !important;
    color: #e8c9a0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid #e8c9a0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.85rem;
    color: #8a4a52 !important;
    background: #f5e6d3;
    border: 1px solid #e8c9a0;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background: #8b1a1a !important;
    color: #fdf6ee !important;
    border-color: #8b1a1a !important;
}

[data-testid="stChatInput"] textarea {
    background: #fff8f0 !important;
    color: #3d0c11 !important;
    border: 2px solid #c8a882 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #c8a882 !important; }
[data-testid="stChatInput"]:focus-within {
    border-color: #8b1a1a !important;
    box-shadow: 0 0 0 3px rgba(139,26,26,0.12) !important;
}

[data-testid="stChatMessage"] {
    background: #fff8f0 !important;
    border: 1px solid #e8c9a0 !important;
    border-radius: 12px !important;
}
[data-testid="stChatMessage"] p { color: #3d0c11 !important; }

[data-testid="stMetric"] {
    background: #fff8f0 !important;
    border: 1px solid #e8c9a0 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] p {
    color: #8a4a52 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: #8b1a1a !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #e8c9a0 !important;
    border-radius: 10px !important;
    background: #fff8f0 !important;
}
[data-testid="stExpander"] summary { color: #8a4a52 !important; font-weight: 500; }

.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: 1px solid #c8a882 !important;
    color: #8b1a1a !important;
    background: #fff8f0 !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #f5e6d3 !important;
    border-color: #8b1a1a !important;
}
.stButton > button[kind="primary"] {
    background: #8b1a1a !important;
    border-color: #8b1a1a !important;
    color: #fdf6ee !important;
}

.stTextArea textarea {
    background: #fff8f0 !important;
    border: 1px solid #e8c9a0 !important;
    border-radius: 8px !important;
    color: #3d0c11 !important;
}

[data-testid="stAlert"] {
    background: #fff8f0 !important;
    border-left: 4px solid #8b1a1a !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p { color: #3d0c11 !important; }

.stCaption, [data-testid="stCaptionContainer"] p {
    color: #8a4a52 !important;
    font-size: 0.78rem !important;
}

.source-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    margin: 2px;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.pdf-pill  { background:#fdf0e0; color:#8b1a1a; border:1px solid #c8a882; }
.docx-pill { background:#e8f0fb; color:#1a3d8b; border:1px solid #a0b8e8; }
.txt-pill  { background:#f0f0f0; color:#555;    border:1px solid #ccc; }
.web-pill  { background:#f0e8fb; color:#5a1a8b; border:1px solid #c0a0e8; }
.text-pill { background:#fdf0e0; color:#8b4a1a; border:1px solid #d4a882; }

.metric-box {
    background: #fff8f0;
    border: 1px solid #e8c9a0;
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-box:hover { transform: translateY(-2px); border-color: #8b1a1a; }
.metric-num { font-size: 2rem; font-weight: 600; margin-bottom: 4px; }
.metric-lbl {
    font-size: 0.72rem; color: #8a4a52; margin-top: 4px;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;
}

.typing-indicator { display:flex; align-items:center; gap:4px; padding:4px 0; }
.typing-dot {
    width:6px; height:6px; border-radius:50%; background:#8b1a1a;
    animation: typing-bounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-bounce {
    0%,60%,100% { transform:translateY(0); opacity:0.4; }
    30%          { transform:translateY(-5px); opacity:1; }
}
.typing-label { font-size:0.78rem; color:#8a4a52; margin-left:4px; font-style:italic; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #fdf6ee; }
::-webkit-scrollbar-thumb { background: #c8a882; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8b1a1a; }

[data-testid="stDownloadButton"] > button {
    background: #3d0c11 !important;
    color: #fdf6ee !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cache models — defined BEFORE they are called ─────────────────────────────
@st.cache_resource
def get_embeddings(): return load_embeddings_model()

@st.cache_resource
def get_reranker():   return load_reranker()

@st.cache_resource
def get_llm():        return load_llm()

# ── Loading screen on first visit ─────────────────────────────────────────────
if "models_loaded" not in st.session_state:
    with st.spinner("🔬 Starting up — loading AI models, please wait ~30s..."):
        embeddings = get_embeddings()
        reranker   = get_reranker()
        llm        = get_llm()
    st.session_state.models_loaded = True
    st.rerun()
else:
    embeddings = get_embeddings()
    reranker   = get_reranker()
    llm        = get_llm()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "vectorstore":   load_vectorstore(embeddings),
    "messages":      [],
    "doc_stats":     {},
    "eval_results":  None,
    "use_agent":     False,
    "use_streaming": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 12px;'>
        <div style='font-family:Playfair Display,serif;font-size:1.25rem;
                    font-weight:600;color:#fdf6ee;'>
            🔬 Research Assistant
        </div>
        <div style='font-size:0.72rem;color:#8a4a52;margin-top:4px;
                    letter-spacing:0.04em;'>
            Multi-source · Agentic · Evaluated
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.session_state.use_agent = st.toggle(
        "🤖 Agentic mode",
        value=st.session_state.use_agent,
        help="Agent decides which tools to use."
    )
    st.session_state.use_streaming = st.toggle(
        "⚡ Streaming mode",
        value=st.session_state.get("use_streaming", True),
        help="Tokens appear as they generate."
    )

    st.markdown("---")
    st.markdown("### Add Sources")

    uploaded = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="PDF · DOCX · TXT"
    )

    url_input = st.text_area(
        "Web URLs (one per line)",
        placeholder="https://example.com/article",
        height=72
    )

    text_input = st.text_area(
        "Paste notes or text",
        placeholder="Paste any text here...",
        height=72
    )

    if st.button("⚙️ Process Sources", type="primary",
                 use_container_width=True):
        pdf_paths   = []
        urls        = [u.strip() for u in url_input.split('\n')
                       if u.strip().startswith('http')]
        text_inputs = [text_input.strip()] if text_input.strip() else []

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
                st.session_state.vectorstore  = vs
                st.session_state.doc_stats    = summary
                st.session_state.messages     = []
                st.session_state.eval_results = None

            for p in pdf_paths:
                try: os.remove(p)
                except: pass

            st.success(f"✅ {len(summary)} source(s) ready!")
            for name, info in summary.items():
                typ = info.get('type', '?')
                st.markdown(
                    f"<span class='source-pill {typ}-pill'>{typ}</span> "
                    f"<span style='font-size:0.75rem;color:#c8a882'>"
                    f"{name[:35]}</span> "
                    f"<span style='font-size:0.7rem;color:#8a4a52'>"
                    f"· {info.get('chunks',0)} chunks</span>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    loaded = get_loaded_sources(st.session_state.vectorstore)
    if loaded:
        st.markdown("### Loaded Sources")
        for src in loaded:
            typ = src['type']
            st.markdown(
                f"<span class='source-pill {typ}-pill'>{typ}</span> "
                f"<span style='font-size:0.75rem;color:#c8a882'>"
                f"{src['name'][:35]}</span>",
                unsafe_allow_html=True
            )

    if st.session_state.messages:
        st.markdown("---")
        if st.button("📄 Export Chat as PDF", use_container_width=True):
            from export import export_chat_to_pdf
            pdf_bytes = export_chat_to_pdf(
                st.session_state.messages,
                get_loaded_sources(st.session_state.vectorstore)
            )
            st.download_button(
                label="⬇️ Download Report",
                data=pdf_bytes,
                file_name=f"research_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("LangChain · ChromaDB · BGE · Groq · RAGAS")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Evaluation", "ℹ️ About"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style='margin-bottom:1rem;padding-top:0.5rem;'>
        <h2 style='font-family:Playfair Display,serif;font-weight:600;
                   font-size:1.7rem;color:#3d0c11;margin:0;'>
            Ask anything about your sources
        </h2>
    </div>
    """, unsafe_allow_html=True)

    mode = "🤖 Agent" if st.session_state.use_agent else \
           "⚡ Streaming" if st.session_state.use_streaming else "Fast"

    if loaded:
        st.caption(
            f"{mode} mode · {len(loaded)} source(s) loaded · "
            f"{len(st.session_state.messages)//2} questions asked"
        )
    else:
        st.info("👈 Upload documents in the sidebar and click "
                "**Process Sources** to begin.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("steps"):
                    with st.expander("🔍 Agent reasoning steps"):
                        for i, step in enumerate(msg["steps"], 1):
                            st.markdown(
                                f"**Step {i}** — `{step['tool']}`  \n"
                                f"Input: {step['input'][:80]}  \n"
                                f"Result: {step['observation'][:120]}"
                            )
                if msg.get("sources"):
                    with st.expander("📎 Sources used"):
                        for s in msg["sources"]:
                            typ = s.get('type', '?')
                            st.markdown(
                                f"<span class='source-pill {typ}-pill'>"
                                f"{typ}</span> **{s['source']}** "
                                f"— *{s['section']}*  \n"
                                f"<small style='color:#8a4a52'>"
                                f"{s['preview']}</small>",
                                unsafe_allow_html=True
                            )
                if msg.get("metrics"):
                    m = msg["metrics"]
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("⏱ Total",      f"{m.get('total_time_s','?')}s")
                    mc2.metric("🔍 Retrieval",  f"{m.get('retrieval_time_s','?')}s")
                    mc3.metric("🤖 Generation", f"{m.get('generation_time_s','?')}s")
                    mc4.metric("💰 Est. cost",  f"${m.get('estimated_cost_usd',0):.6f}")

    # Chat input
    if prompt := st.chat_input(
        "Ask anything about your sources...",
        disabled=not loaded
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            if st.session_state.use_agent:
                st.markdown("""
                <div class='typing-indicator'>
                    <div class='typing-dot'></div>
                    <div class='typing-dot'></div>
                    <div class='typing-dot'></div>
                    <span class='typing-label'>Agent thinking...</span>
                </div>""", unsafe_allow_html=True)
                result = run_agent(
                    prompt,
                    st.session_state.messages[:-1],
                    st.session_state.vectorstore,
                    reranker, llm
                )
                st.markdown(result["answer"])

            elif st.session_state.use_streaming:
                with st.spinner("Retrieving relevant sources..."):
                    stream_result = answer_stream(
                        prompt,
                        st.session_state.messages[:-1],
                        st.session_state.vectorstore,
                        reranker, llm
                    )
                t_gen     = time.time()
                full_text = st.write_stream(stream_result["stream"])
                generation_time = round(time.time() - t_gen, 2)
                result = {
                    "answer":  full_text,
                    "sources": stream_result["sources"],
                    "chunks":  stream_result["chunks"],
                    "steps":   [],
                    "metrics": {
                        **stream_result["metrics"],
                        "generation_time_s": generation_time,
                        "total_time_s": round(
                            stream_result["metrics"].get("retrieval_time_s", 0)
                            + generation_time, 2
                        )
                    }
                }

            else:
                with st.spinner("Searching..."):
                    result = answer(
                        prompt,
                        st.session_state.messages[:-1],
                        st.session_state.vectorstore,
                        reranker, llm
                    )
                st.markdown(result["answer"])

            if result.get("metrics"):
                m = result["metrics"]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("⏱ Total",      f"{m.get('total_time_s','?')}s")
                mc2.metric("🔍 Retrieval",  f"{m.get('retrieval_time_s','?')}s")
                mc3.metric("🤖 Generation", f"{m.get('generation_time_s','?')}s")
                mc4.metric("💰 Est. cost",  f"${m.get('estimated_cost_usd',0):.6f}")

            if result.get("steps"):
                with st.expander("🔍 Agent reasoning steps"):
                    for i, step in enumerate(result["steps"], 1):
                        st.markdown(
                            f"**Step {i}** — `{step['tool']}`  \n"
                            f"Input: {step['input'][:80]}  \n"
                            f"Result: {step['observation'][:120]}"
                        )

            if result.get("sources"):
                with st.expander("📎 Sources used"):
                    for s in result["sources"]:
                        typ = s.get('type', '?')
                        st.markdown(
                            f"<span class='source-pill {typ}-pill'>"
                            f"{typ}</span> **{s['source']}** "
                            f"— *{s['section']}*  \n"
                            f"<small style='color:#8a4a52'>"
                            f"{s['preview']}</small>",
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "steps":   result.get("steps", []),
            "metrics": result.get("metrics", {})
        })

# ── Tab 2: Evaluation ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <h2 style='font-family:Playfair Display,serif;color:#3d0c11;
               font-weight:600;margin-bottom:0.5rem;'>
        RAGAS Evaluation Dashboard
    </h2>
    """, unsafe_allow_html=True)
    st.caption("Measure your RAG system quality with objective metrics.")

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
        st.markdown("""
        <div style='background:#fff8f0;border:1px solid #e8c9a0;
                    border-radius:12px;padding:14px;margin-top:24px;'>
            <div style='font-size:0.78rem;font-weight:500;
                        color:#3d0c11;margin-bottom:10px;'>
                What each metric means
            </div>
            <div style='font-size:0.75rem;color:#8a4a52;line-height:2;'>
                🟢 <b>Faithfulness</b> — no hallucination<br>
                🟡 <b>Answer Relevancy</b> — on-topic<br>
                🔵 <b>Context Precision</b> — clean retrieval<br>
                🟣 <b>Context Recall</b> — found everything<br><br>
                <span style='color:#c8a882;font-size:0.7rem;'>
                    Good scores: &gt; 0.7
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶ Run Evaluation", type="primary", disabled=not loaded):
        questions = [q.strip() for q in eval_questions.split('\n') if q.strip()]
        truths    = [t.strip() for t in eval_truths.split('\n') if t.strip()]

        if len(questions) != len(truths):
            st.error("Questions and ground truths must match in number.")
        else:
            with st.spinner(
                f"Evaluating {len(questions)} questions — 2-4 minutes..."
            ):
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
                        color = "#3d6b1a" if score > 0.7 \
                                else "#8b5a1a" if score > 0.5 \
                                else "#8b1a1a"
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-num' style='color:{color}'>"
                            f"{icon} {score:.2f}</div>"
                            f"<div class='metric-lbl'>{name}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-num' "
                            f"style='color:#8a4a52'>N/A</div>"
                            f"<div class='metric-lbl'>{name}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <h2 style='font-family:Playfair Display,serif;color:#3d0c11;
               font-weight:600;margin-bottom:1rem;'>
        Multi-Source AI Research Assistant
    </h2>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#fff8f0;border:1px solid #e8c9a0;
                    border-radius:12px;padding:16px;margin-bottom:12px;'>
            <div style='font-size:0.78rem;font-weight:500;color:#8b1a1a;
                        text-transform:uppercase;letter-spacing:0.05em;
                        margin-bottom:8px;'>What this does</div>
            <div style='font-size:0.82rem;color:#3d0c11;line-height:1.8;'>
                Upload PDFs, add web URLs, paste text notes — ask questions
                across all sources at once. Every answer cites which source
                and section it came from.
            </div>
        </div>
        <div style='background:#fff8f0;border:1px solid #e8c9a0;
                    border-radius:12px;padding:16px;'>
            <div style='font-size:0.78rem;font-weight:500;color:#8b1a1a;
                        text-transform:uppercase;letter-spacing:0.05em;
                        margin-bottom:8px;'>RAG Techniques</div>
            <div style='font-size:0.82rem;color:#3d0c11;line-height:2;'>
                · Section-aware chunking<br>
                · BGE embeddings (768-dim)<br>
                · Cross-encoder re-ranking<br>
                · MMR retrieval<br>
                · Question-type routing<br>
                · Conversation memory<br>
                · Streaming responses<br>
                · Agentic tool selection<br>
                · RAGAS evaluation
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:#fff8f0;border:1px solid #e8c9a0;
                    border-radius:12px;padding:16px;margin-bottom:12px;'>
            <div style='font-size:0.78rem;font-weight:500;color:#8b1a1a;
                        text-transform:uppercase;letter-spacing:0.05em;
                        margin-bottom:8px;'>Tech Stack</div>
            <div style='font-size:0.82rem;color:#3d0c11;line-height:2;'>
                · LangChain + LangGraph<br>
                · ChromaDB (versioned)<br>
                · BAAI/bge-base-en-v1.5<br>
                · ms-marco cross-encoder<br>
                · Groq · Llama 3.3 70B<br>
                · RAGAS evaluation<br>
                · ReportLab PDF export<br>
                · Streamlit
            </div>
        </div>
        <div style='background:#fff8f0;border:1px solid #e8c9a0;
                    border-radius:12px;padding:16px;'>
            <div style='font-size:0.78rem;font-weight:500;color:#8b1a1a;
                        text-transform:uppercase;letter-spacing:0.05em;
                        margin-bottom:8px;'>Evaluation Results</div>
            <div style='font-size:0.82rem;color:#3d0c11;line-height:2;'>
                🟢 Faithfulness &nbsp;&nbsp;&nbsp; 0.93<br>
                🟡 Ans. Relevancy &nbsp; 0.74<br>
                🔵 Ctx. Precision &nbsp;&nbsp; 0.85<br>
                🟣 Ctx. Recall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1.00
            </div>
        </div>
        """, unsafe_allow_html=True)