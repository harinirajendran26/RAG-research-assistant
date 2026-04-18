import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from sentence_transformers import CrossEncoder

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
duck = DuckDuckGoSearchRun()

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
print(f"Loaded DB: {latest_path}\n")


# ── Tool functions ────────────────────────────────────────────────────────────
def search_documents(query: str) -> str:
    candidates = vectorstore.similarity_search(query, k=15)
    if not candidates:
        return "No relevant content found in the loaded documents."
    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [doc for _, doc in scored[:3]]
    result = ""
    for i, doc in enumerate(top, 1):
        src = doc.metadata.get('source', 'unknown')
        sec = doc.metadata.get('section', '?')
        result += f"\n[Source {i}: {src}, {sec}]\n{doc.page_content}\n"
    return result


def search_web(query: str) -> str:
    try:
        return duck.run(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"


def calculate(expression: str) -> str:
    try:
        allowed = {"__builtins__": {}, "abs": abs, "round": round,
                   "min": min, "max": max, "pow": pow}
        result  = eval(expression, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "search_documents": search_documents,
    "search_web":       search_web,
    "calculator":       calculate
}

TOOL_DESCRIPTIONS = """
You have 3 tools available. To use a tool respond with EXACTLY this format:
TOOL: tool_name
INPUT: your input here

Available tools:
- search_documents: Search loaded PDF documents. Use for questions about research papers or document content.
- search_web: Search the internet. Use for recent news or current events not in documents.
- calculator: Do math. Use for numerical calculations. Input a Python expression like '28.4 * 1.15'.

When you have enough information to answer, respond with:
FINAL: your complete answer here

Rules:
- Always use at least one tool before giving a FINAL answer
- Always cite sources in your FINAL answer
- You can use multiple tools if needed
"""


# ── Manual ReAct agent ────────────────────────────────────────────────────────
def ask_agent(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Build conversation history
    conversation = [
        SystemMessage(content=TOOL_DESCRIPTIONS),
        HumanMessage(content=f"Question: {question}\n\nThink step by step. Which tool should you use first?")
    ]

    max_steps = 5
    step      = 0

    while step < max_steps:
        step += 1
        print(f"\n[Step {step}] Agent thinking...")

        response = llm.invoke(conversation)
        text     = response.content.strip()
        print(f"Agent: {text[:200]}...")

        # Check if agent wants to use a tool
        if "TOOL:" in text and "INPUT:" in text:
            try:
                tool_line  = [l for l in text.split('\n') if l.startswith('TOOL:')][0]
                input_line = [l for l in text.split('\n') if l.startswith('INPUT:')][0]
                tool_name  = tool_line.replace('TOOL:', '').strip()
                tool_input = input_line.replace('INPUT:', '').strip()

                print(f"  → Using tool  : {tool_name}")
                print(f"  → With input  : {tool_input}")

                # Execute the tool
                if tool_name in TOOLS:
                    observation = TOOLS[tool_name](tool_input)
                    print(f"  → Observation : {observation[:150]}...")
                else:
                    observation = f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"

                # Add agent response and tool result to conversation
                conversation.append(response)
                conversation.append(HumanMessage(
                    content=f"Tool result:\n{observation}\n\nContinue. Do you need another tool or do you have enough to give a FINAL answer?"
                ))

            except Exception as e:
                print(f"  → Tool parsing error: {e}")
                conversation.append(response)
                conversation.append(HumanMessage(
                    content="Please use the exact format: TOOL: name\\nINPUT: value\\nOr FINAL: answer"
                ))

        # Check if agent has final answer
        elif "FINAL:" in text:
            final_answer = text.split("FINAL:")[-1].strip()
            print(f"\n── FINAL ANSWER ──────────────────────────────────────")
            print(final_answer)
            return final_answer

        else:
            # Agent responded but didn't use correct format
            conversation.append(response)
            conversation.append(HumanMessage(
                content="Please respond with either TOOL:/INPUT: to use a tool, or FINAL: to give your answer."
            ))

    # Max steps reached
    print("\n── ANSWER (max steps reached) ────────────────────────")
    print(response.content)
    return response.content


# ── Run tests ─────────────────────────────────────────────────────────────────
ask_agent(
    "What BLEU score did the Transformer achieve on "
    "English to German translation?"
)

ask_agent("What are the latest LLM releases in 2025?")

ask_agent(
    "The Transformer achieved 28.4 BLEU on English-German. "
    "If a new model improves this by 15 percent what is the new BLEU score?"
)

# ── Interactive mode ──────────────────────────────────────────────────────────
print("\n── Agent interactive mode. Type 'quit' to exit ──\n")
while True:
    user_input = input("Your question: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    if not user_input:
        continue
    ask_agent(user_input)