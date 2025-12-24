# from dotenv import load_dotenv
# import os
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, Annotated, Sequence
# from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
# from operator import add as add_messages
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.tools import tool

# load_dotenv()

# llm = ChatOpenAI(
#     model="gpt-4o", temperature = 0) # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 

# # Our Embedding Model - has to also be compatible with the LLM
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
# )


# pdf_path = "Stock_Market_Performance_2024.pdf"


# # Safety measure I have put for debugging purposes :)
# if not os.path.exists(pdf_path):
#     raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# # Checks if the PDF is there
# try:
#     pages = pdf_loader.load()
#     print(f"PDF has been loaded and has {len(pages)} pages")
# except Exception as e:
#     print(f"Error loading PDF: {e}")
#     raise

# # Chunking Process
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )


# pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

# persist_directory = r"C:\Vaibhav\LangGraph_Book\LangGraphCourse\Agents"
# collection_name = "stock_market"

# # If our collection does not exist in the directory, we create using the os command
# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)


# try:
#     # Here, we actually create the chroma database using our embeddigns model
#     vectorstore = Chroma.from_documents(
#         documents=pages_split,
#         embedding=embeddings,
#         persist_directory=persist_directory,
#         collection_name=collection_name
#     )
#     print(f"Created ChromaDB vector store!")
    
# except Exception as e:
#     print(f"Error setting up ChromaDB: {str(e)}")
#     raise


# # Now we create our retriever 
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5} # K is the amount of chunks to return
# )

# @tool
# def retriever_tool(query: str) -> str:
#     """
#     This tool searches and returns the information from the Stock Market Performance 2024 document.
#     """

#     docs = retriever.invoke(query)

#     if not docs:
#         return "I found no relevant information in the Stock Market Performance 2024 document."
    
#     results = []
#     for i, doc in enumerate(docs):
#         results.append(f"Document {i+1}:\n{doc.page_content}")
    
#     return "\n\n".join(results)


# tools = [retriever_tool]

# llm = llm.bind_tools(tools)

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]


# def should_continue(state: AgentState):
#     """Check if the last message contains tool calls."""
#     result = state['messages'][-1]
#     return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


# system_prompt = """
# You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
# Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
# If you need to look up some information before asking a follow up question, you are allowed to do that!
# Please always cite the specific parts of the documents you use in your answers.
# """


# tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# # LLM Agent
# def call_llm(state: AgentState) -> AgentState:
#     """Function to call the LLM with the current state."""
#     messages = list(state['messages'])
#     messages = [SystemMessage(content=system_prompt)] + messages
#     message = llm.invoke(messages)
#     return {'messages': [message]}


# # Retriever Agent
# def take_action(state: AgentState) -> AgentState:
#     """Execute tool calls from the LLM's response."""

#     tool_calls = state['messages'][-1].tool_calls
#     results = []
#     for t in tool_calls:
#         print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
#         if not t['name'] in tools_dict: # Checks if a valid tool is present
#             print(f"\nTool: {t['name']} does not exist.")
#             result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
#         else:
#             result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
#             print(f"Result length: {len(str(result))}")
            

#         # Appends the Tool Message
#         results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

#     print("Tools Execution Complete. Back to the model!")
#     return {'messages': results}


# graph = StateGraph(AgentState)
# graph.add_node("llm", call_llm)
# graph.add_node("retriever_agent", take_action)

# graph.add_conditional_edges(
#     "llm",
#     should_continue,
#     {True: "retriever_agent", False: END}
# )
# graph.add_edge("retriever_agent", "llm")
# graph.set_entry_point("llm")

# rag_agent = graph.compile()


# def running_agent():
#     print("\n=== RAG AGENT===")
    
#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break
            
#         messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

#         result = rag_agent.invoke({"messages": messages})
        
#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)


# running_agent()

from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# GROQ LLM (ONLY)
# -------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
)

# -------------------------------------------------
# OPEN-SOURCE EMBEDDINGS (REQUIRED FOR RAG)
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# LOAD PDF
# -------------------------------------------------
pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Loaded PDF with {len(pages)} pages")

# -------------------------------------------------
# SPLITTING
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

# -------------------------------------------------
# VECTOR STORE
# -------------------------------------------------
persist_directory = "./chroma_stock_market"
collection_name = "stock_market"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# -------------------------------------------------
# RETRIEVER TOOL
# -------------------------------------------------
@tool
def retriever_tool(query: str) -> str:
    """
    Search Stock Market Performance 2024 document and return relevant sections.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the document."

    response = []
    for i, doc in enumerate(docs):
        response.append(
            f"Source {i+1}:\n{doc.page_content}"
        )

    return "\n\n".join(response)


tools = [retriever_tool]
llm = llm.bind_tools(tools)

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
system_prompt = """
You are an expert financial research assistant.
Answer questions strictly using the provided document.
Use the retriever tool whenever factual information is required.
Always cite the document content used.
"""

tools_dict = {tool.name: tool for tool in tools}

# -------------------------------------------------
# LLM NODE
# -------------------------------------------------
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

# -------------------------------------------------
# TOOL EXECUTION NODE
# -------------------------------------------------
def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        print(f"Calling tool: {tool_name}")

        if tool_name not in tools_dict:
            result = "Invalid tool."
        else:
            result = tools_dict[tool_name].invoke(tool_args["query"])

        results.append(
            ToolMessage(
                tool_call_id=call["id"],
                name=tool_name,
                content=result
            )
        )

    return {"messages": results}

# -------------------------------------------------
# ROUTING
# -------------------------------------------------
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    return hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0

# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever", False: END}
)

graph.add_edge("retriever", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# -------------------------------------------------
# RUN LOOP
# -------------------------------------------------
def running_agent():
    print("\n=== GROQ RAG AGENT ===")

    while True:
        user_input = input("\nAsk a question (type exit to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = rag_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]}
        )

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


running_agent()
