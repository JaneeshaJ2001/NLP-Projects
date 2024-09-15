This repository contains projects on NLP.

# 1. Q&A RAG App With Gemma And Groq API.

This project is an End-to-End Document Question Answering (Q&A) Retrieval-Augmented Generation (RAG) Application built using the Gemma Model and Groq API. It allows users to upload documents, particularly PDFs, and ask questions based on the document's content. The app returns accurate and contextually relevant answers using a combination of retrieval techniques and generative AI models.

## Features

- Document Ingestion and Chunking: The application reads PDF documents from a directory, processes them into manageable text chunks, and generates embeddings for efficient retrieval.
    - Documents are loaded from the ./pdfs directory using PyPDFDirectoryLoader.
    - The RecursiveCharacterTextSplitter splits the documents into smaller text chunks for more effective processing and retrieval.

- Google Generative AI Embeddings: Utilizes Google's Generative AI Embeddings to convert document chunks into vector representations for semantic understanding and retrieval.

- Vector Store for Document Search: Implements FAISS (Facebook AI Similarity Search) to create embeddings from the document chunks, enabling fast and accurate retrieval of relevant chunks based on semantic similarity.

- Retrieval-Augmented Generation (RAG): The app combines document retrieval with generative AI to answer user questions based on the content of the uploaded documents. The relevant document chunks are retrieved and passed to the language model for answer generation.

- Language Model Integration: Uses ChatGroq, powered by the Llama3-8b-8192 model, to generate human-like responses to user queries.
    - The Groq API, powered by Llama3-8b-8192, produces precise answers based on the context of the retrieved document chunks.
    - The language model is guided by a custom prompt template (ChatPromptTemplate) designed to provide contextually accurate answers.

- Response Time Monitoring: The app tracks and displays the time taken to process and respond to each query, allowing for performance evaluation.


## Getting Started
To get the app running, follow these steps:

1. Install the required dependencies:
```plaintext
    pip install -r requirements.txt
```
2. Set up environment variables: Ensure you have a .env file in the root directory with the following API keys:
```plaintext
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
```
3. Place your PDF documents inside the ./pdfs directory.
4. Run the Streamlit app: Start the app using the following command:
```plaintext
    streamlit run app.py
```
5. Initialize the document embeddings by clicking the "Initialize Documents Embedding" button in the app.
6. Ask a question by entering it into the input box and view the results in real time!