import os 
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
# Message state is basically a object the holds thge entire history of messages the happended during we run the graph 
# it dynamcally keeps appending as we run 
# @ tool is tool we know that much
# so when llm decides to call it 
# toolNode excures and returns output to llm 
from react import llm , tools
load_dotenv()
system_message = """
You are a helpful assistant that can use tools 
to answer questions 
"""

def run_agent_reasoning(state:MessagesState)->MessagesState:#type:ignore 
    """
    Run the agent reasoning node 
    """
    response = llm.invoke([{"role" : "system","content":system_message},*state["messages"]])
    return {"messages":[response]}

tool_node = ToolNode(tools)