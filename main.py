import os 
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END
from nodes import run_agent_reasoning , tool_node
load_dotenv()
Agent_reason = "agent_reason"
ACT = 'act'
last =-1 

def should_continue(state:MessagesState)->str:#type:ignore
    if not state["messages"][last].tool_calls:#type:ignore
        return END 
    return ACT
# MessagesState is ONE shared state for the entire graph execution.
# Nodes do not have their own states.
# Each node receives the latest graph state, returns updates,
# and LangGraph merges those updates into the shared state.
# The next node sees the updated state.
flow = StateGraph(MessagesState)
# create a graph blueprint/workflow
# MessagesState defines the shared state schema
# every node receives this state and returns updates to it
# state mainly contains the conversation history (messages)
# nodes do NOT have separate states; they all work on this shared state
flow.add_node(Agent_reason, run_agent_reasoning)
flow.set_entry_point(Agent_reason)
flow.add_node(ACT,tool_node)
flow.add_conditional_edges(Agent_reason, should_continue, {
    END:END,
    ACT:ACT})
flow.add_edge(ACT, Agent_reason)

app = flow.compile()
#takes the blueprint and creates an executable graph.
app.get_graph().draw_mermaid_png(output_file_path="flow.png")
if __name__ == "__main__":
   print("Hello ReAct LangGraph with Function Calling")
   res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
   print(res["messages"][last].content)