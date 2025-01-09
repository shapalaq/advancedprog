import streamlit as st
import logging
import time
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import PyPDF2  # Для обработки PDF

logging.basicConfig(level=logging.INFO)

def read_file(file):
    """
    Функция для чтения содержимого файлов (PDF или TXT).
    """
    if file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logging.error(f"Ошибка при чтении PDF: {str(e)}")
            return "Не удалось прочитать PDF файл."
    elif file.name.endswith('.txt'):
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            logging.error(f"Ошибка при чтении TXT: {str(e)}")
            return "Не удалось прочитать текстовый файл."
    else:
        return "Формат файла не поддерживается. Загрузите PDF или TXT."

def stream_chat(model, messages, temperature=0.7, max_tokens=256):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        response = ""
        response_placeholder = st.empty()

        resp = llm.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        for r in resp:
            response += r.delta
            response_placeholder.write(response)

        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("Chatbot с анализом файлов")
    logging.info("Приложение запущено")

    model = st.sidebar.selectbox("Выберите модель", ["llama3.1:8b", "llama3.2:1b", "all-minilm:latest"])
    logging.info(f"Модель выбрана: {model}")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Ты — умный ассистент, который может анализировать текстовые файлы."}
        ]

    uploaded_file = st.sidebar.file_uploader("Загрузите файл (PDF или TXT)", type=["pdf", "txt"])
    if uploaded_file:
        file_content = read_file(uploaded_file)
        if file_content.startswith("Формат файла не поддерживается"):
            st.sidebar.error(file_content)
        else:
            st.sidebar.success("Файл успешно загружен.")
            st.session_state.messages.append({"role": "system", "content": f"Вот содержимое файла:\n\n{file_content}"})

    if prompt := st.chat_input("Введите свой вопрос"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"Ввод пользователя: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Генерация ответа...")

                with st.spinner("Пишу..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        st.session_state.messages.append({"role": "assistant", "content": response_message})
                        st.write(f"Ответ сгенерирован за {duration:.2f} секунд.")
                        logging.info(f"Ответ: {response_message}, Время: {duration:.2f} секунд")
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("Произошла ошибка при генерации ответа.")
                        logging.error(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
