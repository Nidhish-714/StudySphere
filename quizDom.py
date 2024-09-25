from educhain import Educhain
from educhain.core import config
from langchain.chat_models import ChatOpenAI
custom_template = """
Generate {num} multiple-choice question (MCQ) based on the given topic and level.
Provide the question, four answer options, and the correct answer.
Topic: {topic}
Learning Objective: {learning_objective}
Difficulty Level: {difficulty_level}
"""

llama = ChatOpenAI(
    model="llama-3.1-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key="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"  # Assuming userdata is a dictionary with the API key
)


llm_config = config.LLMConfig(
    custom_model=llama
)

client = Educhain(llm_config)

result = client.qna_engine.generate_questions(
    topic="Python Programming",
    num=10,
    learning_objective="Usage of Python classes",
    difficulty_level="Hard",
    prompt_template=custom_template,
)

result.show()