from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

app = Flask(__name__)

# Global variables to hold the vector store and conversation chain
vector_store = None
conversation_chain = None
history = []

# Create a prompt template
template = """
You are an assistant specialized in analyzing documents. The user will provide questions based on a document.
Please answer the following question using the context provided:

Context: {context}

Question: {question}

Answer in a concise and informative manner.
Try to answer in bullet points
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Function to create the conversational chain
def create_conversational_chain(vector_store):
    groq_api_key = os.getenv('GROQ_API_KEY')
    model = 'mixtral-8x7b-32768'

    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

# Endpoint to upload files and process them
@app.route('/upload', methods=['POST'])
def upload_files():
    global vector_store, conversation_chain
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    text = []
    for file in files:
        file_extension = os.path.splitext(file.filename)[1]
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

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the conversation chain
    conversation_chain = create_conversational_chain(vector_store)

    return jsonify({"message": "Documents processed successfully"}), 200

# Endpoint to handle chat
@app.route('/chat', methods=['POST'])
def chat():
    global history, conversation_chain
    if not conversation_chain:
        return jsonify({"error": "No documents processed. Please upload documents first."}), 400

    query = request.json.get('question')
    if not query:
        return jsonify({"error": "No question provided"}), 400

    # Retrieve relevant context from the retriever first
    context = conversation_chain.retriever.get_relevant_documents(query)

    # Format the input with the context using the prompt template
    formatted_prompt = prompt.format(context=context, question=query)
    
    # Pass the formatted prompt to the LLM within the chain
    result = conversation_chain({"question": formatted_prompt, "chat_history": history})

    # Append to the conversation history
    history.append((query, result["answer"]))

    return jsonify({"answer": result["answer"]}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
