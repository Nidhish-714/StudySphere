import os
from dotenv import load_dotenv
import tempfile
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
import streamlit as st
from educhain import Educhain, LLMConfig
# from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Prompt Template for Code 1
template = """
You are an assistant specialized in analyzing documents. Your work is to provide topic of that document. 
Topic : 
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

def conversation_chat(query, chain, history):
    context = chain.retriever.get_relevant_documents(query)
    formatted_prompt = prompt.format(context=context, question=query)
    result = chain({"question": formatted_prompt, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def create_conversational_chain(vector_store):
    groq_api_key = os.getenv('GROQ_API_KEY')
    model = 'mixtral-8x7b-32768'

    # Initialize LLM for document chat (Code 1)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the chain for document topic extraction
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def extract_document_topic(uploaded_files):
    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    chain = create_conversational_chain(vector_store)

    # Assume we pass a simple query to extract the document topic.
    query = "What is the topic of this document?"
    history = []
    document_topic = conversation_chat(query, chain, history)
    
    return document_topic

def generate_questions_based_on_topic(topic):
    # Initialize the Google Generative AI model (Code 2)
    gemini_flash =groq_llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model='mixtral-8x7b-32768'
    )


    flash_config = LLMConfig(custom_model=gemini_flash)
    client = Educhain(flash_config)

    # Generate questions based on the document topic
    ques = client.qna_engine.generate_questions(topic=topic, num=10)
    return ques

def main():
    # Simulate file uploads (In actual case, this would be through an upload function)
    uploaded_files  = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)  # Replace with actual file upload logic

    if uploaded_files:
        # Extract the document topic from Code 1
        document_topic = extract_document_topic(uploaded_files)
        print(f"Extracted Document Topic: {document_topic}")

        # Use the extracted topic in Code 2 to generate questions
        questions = generate_questions_based_on_topic(document_topic)
        print(f"Generated Questions: {questions.json()}")

if __name__ == "__main__":
    main()
