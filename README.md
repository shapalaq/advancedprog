# Title
Chat with LLMs Models

# Description
This project is a chat application that allows users to interact with language models (LLMs) like Ollama. The application enables users to ask questions, receive answers, and store messages (including embeddings) in a Chromadb database. The app also supports file uploads, allowing users to load PDF and TXT files and ask questions based on their content. Chat history is preserved even if the app is restarted, thanks to integration with Chromadb.

# Features

Interaction with models through streaming (real-time).
File Upload: Upload and process PDF and TXT files to ask questions about their content.
Storing chat history (questions and answers) and embeddings in a database.
Retrieving chat history upon app startup.
Support for multiple models, such as llama3.2:1b.
Storing embeddings to improve search quality and user interaction.

1. Install Python
Ensure that Python 3.7 or later is installed. You can check the installed version using the following command:

bash
python --version

2. Install Dependencies
streamlit — for displaying the interface.
llama_index — for working with models and chat functionality.
chromadb — for storing messages and embeddings.
ollama — for generating embeddings and interacting with models.
PyPDF2 — for extracting text from PDF files.

bash
git clone <repository_url>
cd <project_folder>
python -m venv venv
source venv/bin/activate  # For macOS/Linux
# For Windows use: venv\Scripts\activate
pip install -r requirements.txt

3. Install and Set Up Ollama
To use the ollama library, you’ll need to install it if it’s not already installed:

bash
pip install ollama
Running the App
After installing all the dependencies, you can run the application using the following command:

bash
streamlit run chat.py
This will open the web application in your browser.

# Project Structure
chat.py: The main file containing the application code.
requirements.txt: A file containing the dependencies needed to run the project.
README.md: This file with instructions on how to use the app.
# How the App Works
Upon starting, the app displays an interface where the user can choose a model.
The user enters a prompt, which is sent to the model.
The model generates a response, which is displayed in the chat.
All messages (questions and responses) are stored in Chromadb along with their embeddings.
The chat history is loaded when the app starts and displayed in the interface.
# Logging
Logs are output to the console and a file, which helps to track errors and processes within the application. Logging enables you to monitor:

Interaction with models.
Errors during response generation.
The duration of request processing.
# Possible Improvements
Adding support for multiple languages.
Expanding search and filtering capabilities for chats by keywords.
Optimizing performance for handling large datasets.
# Contributors
[Daniyal Assanov or shalapaq]
# License
This project is licensed under the MIT License.

