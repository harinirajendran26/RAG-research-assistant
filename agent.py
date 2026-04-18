from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from rag_engine import retrieve, generate

duck = DuckDuckGoSearchRun()

TOOL_DESCRIPTIONS = """
You have 4 tools. Respond with EXACTLY this format to use one:
TOOL: tool_name
INPUT: your input

Tools:
- search_sources: Search loaded PDFs, URLs, and text notes
- search_web: Search internet for recent/live information
- calculator: Math. Input a Python expression like '28.4 * 1.15'
- direct_answer: Answer from conversation context only (no search needed)

When ready to give final answer:
FINAL: your complete answer with source citations
"""


def run_agent(
    question: str,
    history: list,
    vectorstore,
    reranker,
    llm
) -> dict:
    """
    Agentic answering — decides which tools to use,
    chains multiple tools if needed.
    """
    conversation = [
        SystemMessage(content=TOOL_DESCRIPTIONS),
        HumanMessage(content=
            f"Question: {question}\n\n"
            f"Think: which tool is most appropriate?"
        )
    ]

    all_chunks   = []
    steps_taken  = []
    max_steps    = 5
    step         = 0

    while step < max_steps:
        step    += 1
        response = llm.invoke(conversation)
        text     = response.content.strip()

        if "TOOL:" in text and "INPUT:" in text:
            try:
                lines      = text.split('\n')
                tool_line  = next(l for l in lines if l.startswith('TOOL:'))
                input_line = next(l for l in lines if l.startswith('INPUT:'))
                tool_name  = tool_line.replace('TOOL:', '').strip()
                tool_input = input_line.replace('INPUT:', '').strip()

                # Execute tool
                if tool_name == "search_sources" and vectorstore:
                    chunks      = retrieve(question, vectorstore, reranker)
                    all_chunks.extend(chunks)
                    observation = "\n".join([
                        f"[{doc.metadata.get('source','?')}] {doc.page_content[:200]}"
                        for doc in chunks
                    ]) or "No relevant content found."

                elif tool_name == "search_web":
                    try:
                        observation = duck.run(tool_input)
                    except Exception as e:
                        observation = f"Web search failed: {e}"

                elif tool_name == "calculator":
                    try:
                        allowed = {"__builtins__": {}, "abs": abs,
                                   "round": round, "pow": pow}
                        result      = eval(tool_input, allowed)
                        observation = f"Result: {result}"
                    except Exception as e:
                        observation = f"Calculation error: {e}"

                elif tool_name == "direct_answer":
                    observation = "Proceeding with direct answer from context."
                else:
                    observation = f"Unknown tool: {tool_name}"

                steps_taken.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "observation": observation[:200]
                })

                conversation.append(response)
                conversation.append(HumanMessage(
                    content=f"Tool result:\n{observation}\n\n"
                            f"Continue. Use another tool or give FINAL answer."
                ))

            except Exception as e:
                conversation.append(response)
                conversation.append(HumanMessage(
                    content=f"Error: {e}. Use TOOL:/INPUT: or FINAL: format."
                ))

        elif "FINAL:" in text:
            final = text.split("FINAL:")[-1].strip()

            # Build sources list
            sources = []
            seen    = set()
            for doc in all_chunks:
                src = doc.metadata.get('source', '?')
                typ = doc.metadata.get('type', '?')
                sec = doc.metadata.get('section', '?')
                key = f"{src}_{sec}"
                if key not in seen:
                    sources.append({
                        "source":  src,
                        "type":    typ,
                        "section": sec,
                        "preview": doc.page_content[:120] + "..."
                    })
                    seen.add(key)

            return {
                "answer":  final,
                "sources": sources,
                "chunks":  all_chunks,
                "steps":   steps_taken
            }

        else:
            conversation.append(response)
            conversation.append(HumanMessage(
                content="Use TOOL:/INPUT: format or FINAL: to answer."
            ))

    # Max steps — return what we have
    return {
        "answer":  response.content,
        "sources": [],
        "chunks":  all_chunks,
        "steps":   steps_taken
    }