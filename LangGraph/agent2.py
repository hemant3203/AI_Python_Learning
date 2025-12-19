import os 
from typing import TypedDict,List,Union
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()


class AgentState(TypedDict):
    message:List[Union[HumanMessage,AIMessage]]

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

def process(state:AgentState)->AgentState:
    response=llm.invoke(state["message"])

    state["message"].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}\n")
    print("CURRENT STATE:",state["message"])
    return state

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()


conversation_history=[]

user_input=input("Enter: ")
while user_input !="exit":
    conversation_history.append(HumanMessage(content=user_input))
    result=agent.invoke({"message":conversation_history})

    print(result["message"])
    conversation_history=result["message"]

    user_input=input("Enter: ")


with open("loggin.txt","w") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("END OF CONVERSATION\n")

print ("Conversation log saved to logging.txt")