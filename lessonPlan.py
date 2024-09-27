from flask import Flask, request, jsonify
from educhain import Educhain
from educhain.core import config
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Replace with your OpenAI API credentials
custom_instructions = "Include real-world examples"  # Predefined instruction

llama = ChatOpenAI(
    model="llama-3.1-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv('GROQ_API_KEY')  # Replace with your actual key
)

llm_config = config.LLMConfig(
    custom_model=llama
)

client = Educhain(llm_config)

app = Flask(__name__)


@app.route("/generate_lesson_plan", methods=["POST"])
def generate_lesson_plan():
    if request.method == "POST":
        data = request.get_json(force=True)  # Get data from the request body

        topic = data.get("topic")

        if not topic:
            return jsonify({"error": "Missing required parameter: topic"}), 400

        try:
            lesson_plan = client.content_engine.generate_lesson_plan(
                topic=topic, custom_instructions=custom_instructions
            )
            return jsonify(lesson_plan.json())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid request method"}), 405


if __name__ == "__main__":
    app.run(debug=True) 