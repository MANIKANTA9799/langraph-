from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
# ChatPromptTemplate -> creates a reusable chat prompt with variables

# MessagesPlaceholder -> inserts a list of existing messages
# (chat history) into the prompt at runtime
from langchain_ollama import ChatOllama

reflection_prompt = ChatPromptTemplate.from_messages([
    # create a ChatPromptTemplate from the list of messages below
    # when invoked, these messages will be assembled into the final prompt

    (
        "system",

        # system message = instructions for the llm
        # tells the model to behave like a viral twitter influencer

        # the model's task:
        # 1. grade the tweet
        # 2. critique the tweet
        # 3. suggest improvements

        # recommendations should include:
        # - tweet length
        # - virality potential
        # - writing style
        # - engagement hooks
        # - clarity and readability

        "You are a viral twitter influencer grading a tweet. "
        "Generate critique and recommendations for the user. "
        "Always provide detailed recommendations, including requests "
        "for length, virality, style, etc."
    ),

    # placeholder for chat history

    # variable_name="messages" means:
    # "when this prompt runs, look for a variable called messages"

    # example:
    # prompt.invoke({
    #     "messages": state["messages"]
    # })

    # LangChain will automatically insert all messages found in
    # state["messages"] at this position in the prompt

    # if state["messages"] contains:
    # HumanMessage("tweet: hello world")
    #
    # then that message gets inserted here

    MessagesPlaceholder(variable_name="messages")

])



generation_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system",
        "You are a Twitter techie influencer assistant tasked with writing excellent Twitter posts."
      "Generate the best Twitter posts possible for the user's request."
    "If the user provides critique, respond with a revised version of your previous attempts"),
    MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOllama( model="gpt-oss:20b",
    temperature=0 )

generate_chain  = generation_prompt|llm
reflect_chain = reflection_prompt|llm