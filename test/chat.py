import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import chromadb
import ollama

logging.basicConfig(level=logging.INFO)

ollama.embeddings(
  model='all-minilm',
  prompt='Llamas are members of the camelid family',
)

client = chromadb.Client()
collection = client.get_or_create_collection(name="docs")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def store_embeddings_in_chromadb(prompt, response):
    try:
        # Create embeddings using ollama
        prompt_embedding = ollama.embeddings(model='all-minilm', prompt=prompt)
        response_embedding = ollama.embeddings(model='all-minilm', prompt=response)

        # Add embeddings to Chromadb collection
        collection.add(
            documents=[prompt, response],
            embeddings=[prompt_embedding, response_embedding],
            metadatas=[{'type': 'prompt'}, {'type': 'response'}],
            ids=[str(time.time()), str(time.time())]  # Use time as unique ids
        )

        logging.info(f"Stored prompt and response embeddings in Chromadb.")
    
    except Exception as e:
        logging.error(f"Error storing embeddings: {str(e)}")

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["llama3.2:1b"])
    logging.info(f"Model selected: {model}")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")
                        store_embeddings_in_chromadb(prompt, response_message_with_duration)
                        
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()