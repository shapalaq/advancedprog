import os
import time
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from io import BytesIO
import tempfile

# Constants
LLM_MODEL = "llama3.1:8b"
BASE_URL = "http://localhost:11434"

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list):
            raise ValueError("Input to the embedding function must be a string or a list of strings.")
        return self.langchain_embeddings.embed_documents(input)

embedding = ChromaDBEmbeddingFunction(
    langchain_embeddings=OllamaEmbeddings(model=LLM_MODEL, base_url=BASE_URL)
)

collection_name = "rag_collection_demo"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "RAG collection for documents"},
    embedding_function=embedding
)

def read_pdf(file):
    """Читает текст из PDF-файла"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    try:
        doc = fitz.open(tmp_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
    finally:
        doc.close()
        os.remove(tmp_file_path)
    return text

def process_and_add_documents(content, file_name_prefix=""):
    """Обрабатывает документ и добавляет в ChromaDB"""
    if not content:
        print(f"Warning: Content from {file_name_prefix} is empty. Skipping processing.")
        return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    if not chunks:
        print(f"No chunks found in the content of {file_name_prefix}")
    else:
        print(f"Splitting content into {len(chunks)} chunks for {file_name_prefix}.")
    chunk_ids = [f"{file_name_prefix}_chunk_{i}" for i in range(len(chunks))]
    if not chunks or not chunk_ids:
        print(f"Warning: No valid chunks or chunk IDs for {file_name_prefix}. Skipping add to collection.")
        return
    collection.add(documents=chunks, ids=chunk_ids)

def add_chat_to_memory(user_query, bot_response):
    """Сохраняем историю чата в ChromaDB"""
    chat_entry = f"User: {user_query} \nAssistant: {bot_response}"
    chat_id = f"chat_{int(time.time())}"
    collection.add(documents=[chat_entry], ids=[chat_id])

def retrieve_chat_memory(n_results=5):
    """Извлекаем последние n сообщений из памяти"""
    results = collection.query(query_texts=[""], n_results=n_results)
    documents = results["documents"] if results else []
    
    # Преобразуем список списков в один список строк
    flat_documents = []
    for sublist in documents:
        if isinstance(sublist, list):
            flat_documents.extend(sublist)
        elif isinstance(sublist, str):
            flat_documents.append(sublist)
    
    return flat_documents

def rag_pipeline(query_text):
    """Обрабатывает запрос с учетом памяти (контекста)"""
    chat_memory = retrieve_chat_memory()
    context = " \n".join(chat_memory) if chat_memory else ""
    
    prompt = f"{context}\nUser: {query_text}\nAssistant:"
    response = query_ollama(prompt)
    
    add_chat_to_memory(query_text, response)
    return response

def query_ollama(prompt):
    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL)
    return llm.invoke(prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("Interactive AI Assistant for the Constitution of Kazakhstan")
    model = st.sidebar.selectbox("Choose a model", [LLM_MODEL])
    
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .pdf files", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            content = read_pdf(uploaded_file)
            process_and_add_documents(content, file_name_prefix=uploaded_file.name)
        st.sidebar.success(f"Uploaded and processed {len(uploaded_files)} file(s).")
    
    prompt = st.chat_input("Ask your question:")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response_message = rag_pipeline(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response_message})
                    st.write(response_message)

if __name__ == "__main__":
    main()


