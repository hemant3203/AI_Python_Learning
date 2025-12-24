from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import re

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

load_dotenv()

# =====================================================
# GLOBAL DOCUMENT STATE
# =====================================================
document_content = ""


# =====================================================
# STATE
# =====================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =====================================================
# HELPER
# =====================================================
def extract_filename(text: str) -> str | None:
    """
    Extract filename from natural language save commands.
    Examples supported:
    - save as file.txt
    - save file.txt
    - save the file with name file.txt
    - save with name file.txt
    """
    text = text.lower()

    # Priority 1: explicit .txt filename anywhere
    match = re.search(r"([\w\-]+\.txt)", text)
    if match:
        return match.group(1)

    # Priority 2: 'with name <filename>'
    match = re.search(r"(?:with name|named)\s+([\w\-]+)", text)
    if match:
        return match.group(1) + ".txt"

    return None


# =====================================================
# TOOLS (USED MANUALLY)
# =====================================================
@tool
def update(content: str) -> str:
    """
    Append new content to the existing document
    and return the full updated document.
    """
    global document_content

    if document_content.strip():
        document_content += "\n\n" + content
    else:
        document_content = content

    return (
        "✅ Document updated successfully.\n\n"
        "📄 CURRENT DOCUMENT\n"
        "----------------------------------\n"
        f"{document_content}\n"
        "----------------------------------"
    )


@tool
def save(filename: str = "document.txt") -> str:
    """
    Save the current document content into a text file.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(document_content)

    return f"✅ Document saved successfully as {filename}"


# =====================================================
# MODEL (NO TOOL BINDING)
# =====================================================
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=900,
)


# =====================================================
# AGENT
# =====================================================
def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
You are Drafter, a professional document writer.

RULES:
- Write LONG, DETAILED content (minimum 300 words when expanding)
- NEVER overwrite existing content
- If user asks to add/update → WRITE CONTENT ONLY
- If user asks to save → respond with exactly: SAVE_DOCUMENT

CURRENT DOCUMENT:
-----------------
{document_content}
-----------------
"""
    )

    if not state["messages"]:
        user_message = HumanMessage(
            content="What document would you like to create or edit?"
        )
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)

    messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(messages)

    # ---------------- SAVE ----------------
    if "SAVE_DOCUMENT" in response.content.upper():
        filename = extract_filename(user_message.content) or "document.txt"

        if not document_content.strip():
            msg = "❌ Cannot save an empty document. Please add content first."
            print(msg)
            return {"messages": state["messages"] + [user_message, AIMessage(content=msg)]}

        result = save.invoke({"filename": filename})
        print(result)

        return {
            "messages": state["messages"]
            + [user_message, AIMessage(content=result)]
        }

    # ---------------- UPDATE ----------------
    if response.content.strip():
        result = update.invoke({"content": response.content})
        print(result)

        return {
            "messages": state["messages"]
            + [user_message, AIMessage(content=result)]
        }

    return {"messages": state["messages"] + [user_message]}


# =====================================================
# ROUTING
# =====================================================
def should_continue(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "saved successfully" in msg.content.lower():
            return "end"
    return "continue"


# =====================================================
# GRAPH
# =====================================================
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


# =====================================================
# RUNNER
# =====================================================
def run_document_agent():
    print("\n===== DRAFTER =====")
    state = {"messages": []}

    for _ in app.stream(state):
        pass

    print("\n===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
