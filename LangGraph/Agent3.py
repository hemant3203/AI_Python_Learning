# React Agent (Reasoning and Acting Agent)
#  start
# Agent  ------>Continue Loop ----->agent again agent 
#  if agent is stop using tools than it will end at last 

# We can create a tools message for this 


from typing import TypedDict,Annotated,Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
# from panel import state



load_dotenv()


# Annotated- provides additional context without affecting the type itself

# email=Annotated[str, "This Has to be a valid email format!"]

# print(email.__metadata__)


# Sequences - TO automatically handle the state updates for sequence such as by adding new messages to a chat history


# Reducer Function 
#Rule that controls how updates from nodes are combined with the exixting state.
# Tells us how to merge new Data into the current state

#without a reducer, updates would have replaced the existing value entirely!


# #Without a reducer
# state={"messages":["Hii"]}
# update={"messages":["Nice to meet you!"]}
# new_state={"messages":["Nice to meet you!"]}

# #with a reducer

# state={"messages":["hii"]}
# update={"messages":["NIce to meet you!"]}
# new_state={"messages":["hii","Nice to meet you!"]}

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

@tool 
def add(a:int,b:int):
    """This is an addition function that adds 2 numers together"""

    return a+b

tools=[add]

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
).bind_tools(tools)


def model_call(State:AgentState)->AgentState:
    System_prompt=SystemMessage(content="you are my AI assistant ,please answer my query to the best of your ability.")
    response=llm.invoke([System_prompt]+list(State["messages"]))
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages=state["messages"]
    last_message=messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph=StateGraph(AgentState)
graph.add_node("our_agent",model_call)


tool_node=ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")


graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","our_agent")
app=graph.compile()

def print_stream(stream):
    for s in stream:
        if "messages" in s:
            last_message = s["messages"][-1]

            if hasattr(last_message, "pretty_print"):
                last_message.pretty_print()
            else:
                print(last_message)


inputs ={"messages":[("user","Add 3 + 4 . Add 6+40 " )]}
print_stream(app.stream(inputs,stream_mode="values"))


