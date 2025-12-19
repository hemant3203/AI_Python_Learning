from typing import Dict,TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    message:str 

def greeting_node(state:AgentState)-> AgentState:
    """Simple node that add a greeting message to the state"""

    state['message']="hey" + state["message"] + ", how is your day going ?"

    return state