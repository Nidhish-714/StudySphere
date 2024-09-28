import os
from dotenv import load_dotenv
import tempfile
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
from educhain import Educhain, LLMConfig
import streamlit as st
from streamlit_chat import message

# Load environment variables
load_dotenv()

# Prompt Template for Document Analysis
template = """
You are an assistant specialized in analyzing documents. Your work is to provide topic of that document.
Topic : 
"""

topic_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Prompt Template for answering document questions
qa_template = """
You are an assistant specialized in analyzing documents. The user will provide questions based on a document.
Please answer the following question using the context provided:

Context: {context}

Question: {question}

Answer in a concise and informative manner.
Try to answer in bullet points
"""
qa_prompt = PromptTemplate(input_variables=["context", "question"], template=qa_template)

def conversation_chat(query, chain, history):
    context = chain.retriever.get_relevant_documents(query)
    formatted_prompt = qa_prompt.format(context=context, question=query)
    result = chain({"question": formatted_prompt, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def create_conversational_chain(vector_store):
    groq_api_key = os.getenv('GROQ_API_KEY')
    model = 'mixtral-8x7b-32768'
    
    # Initialize Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversational chain
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
        elif file_extension in [".docx", ".doc"]:
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
    
    query = "What is the topic of this document?"
    history = []
    document_topic = conversation_chat(query, chain, history)
    
    return document_topic, chain

def generate_questions_based_on_topic(topic):
    groq_llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model='mixtral-8x7b-32768'
    )
    flash_config = LLMConfig(custom_model=groq_llm)
    client = Educhain(flash_config)

    # Generate questions based on the document topic
    questions = client.qna_engine.generate_questions(topic=topic, num=10)
    return questions

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def main():
    load_dotenv()
    initialize_session_state()
    
    st.title("Document ChatBot with Quiz Generation :books:")
    st.sidebar.title("Document Processing")
    
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
    if uploaded_files:
        # Extract the document topic and chain for chat
        document_topic, chain = extract_document_topic(uploaded_files)
        st.write(f"Extracted Document Topic: {document_topic}")
        
        # Let the user choose between Chat or Quiz
        option = st.radio("What would you like to do?", ("Chat with the document", "Generate a quiz"))

        if option == "Chat with the document":
            display_chat_history(chain)
        
        elif option == "Generate a quiz":
            # Generate questions based on the document topic
            questions = generate_questions_based_on_topic(document_topic)
            st.write(f"Generated Questions: {questions.json()}")

if __name__ == "__main__":
    main()
