from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion,ReviseAnswer
# so this tool node does something very important fir us 
# it basically does check if therir is any tool call need to be done 
# from checking the state then does it if needed returns result 
tavily_tool = TavilySearch(max_results = 5)

# so here we will have 2 tools one to do the intail search other to modify we can 
# acheviev this functionality with one only but this makes debugging easier and 
# visibilty too

def run_queries(search_queries:list[str], **kwargs):
    """Run the generated queries """
    return tavily_tool.batch([{"query":query} for query in search_queries])

excecute_tools = ToolNode(
[
StructuredTool.from_function(run_queries, name = AnswerQuestion.__name__),
StructuredTool.from_function(run_queries, name = ReviseAnswer.__name__),
]

)
# structured tool does this 
#Take this Python function
#and convert it into a LangChain tool.


load_dotenv()



