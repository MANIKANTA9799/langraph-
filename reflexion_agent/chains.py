import datetime 
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from schemas import AnswerQuestion,ReviseAnswer
load_dotenv()
# ollama model
llm = ChatOllama(
    model="gpt-oss:120b",
    temperature=0
)


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.

Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.
""",
        ),

        # inserts conversation history from input["messages"]
        MessagesPlaceholder(variable_name="messages"),

        (
            "system",
            "Answer the user's question above using the required format.",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer"
)

first_responder = (
    first_responder_prompt_template|llm.with_structured_output(AnswerQuestion)
)

revise_instructions = """
Revise your previous answer using the new information.

- Use the previous critique to improve the answer.
- Add important missing information.
- Remove unnecessary information.
- Include numerical citations.
- Add a References section.

Format:

References
[1] https://example.com
[2] https://example.com

Keep the answer under 250 words.
"""


revisor = (
    actor_prompt_template.partial(
        first_instruction=revise_instructions
    )
    | llm.with_structured_output(ReviseAnswer)
)
