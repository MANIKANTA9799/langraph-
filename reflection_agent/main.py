import os 
from dotenv import load_dotenv
from typing import TypedDict, Annotated
# TypedDict:
# used to define the structure (schema) of a dictionary
# helps type checkers understand what keys and value types are expected
# commonly used for LangGraph state definitions

# Annotated:
# adds extra metadata to a type
# LangGraph uses this metadata to define how state fields should be updated
# for example, how messages should be merged across nodes
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain
load_dotenv()
reflect = "reflect"
generate = " generate"

class MessageGraph(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def generation_node(state:MessageGraph):
    return {"messages":[generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state:MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(generate, generation_node)
builder.add_node(reflect, reflection_node)
builder.set_entry_point(generate)

def should_continue(state:MessageGraph):
    if(len(state["messages"])>6):
        return END
    return reflect


builder.add_conditional_edges(generate, should_continue)
builder.add_edge(reflect, generate)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()



if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)#type:ignore 
    print(response)