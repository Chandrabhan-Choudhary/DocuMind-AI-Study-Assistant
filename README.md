# 🧠 DocuMind: AI Study Assistant

> **Turn static textbooks into interactive study partners using RAG and Gemini 1.5 Pro.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/GenAI-Gemini%201.5%20Pro-orange)
![Framework](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![DB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple)

## 📖 Overview
**DocuMind** is an AI-powered web application designed to solve the problem of passive studying. Instead of just reading a PDF, DocuMind allows students to chat with their study material.

Built with **Streamlit**, it provides a clean UI where users can upload any PDF textbook. The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to read the document, generate embeddings, and provide context-aware answers to student questions.

## ⚙️ Architecture
The system is built on a custom RAG pipeline:
1.  **Ingestion:** Loads PDF documents using `PyPDFLoader`.
2.  **Storage:** Embeds text using Google GenAI Embeddings and stores it in **ChromaDB**.
3.  **Retrieval:** Uses Similarity Search to find relevant context.
4.  **Generation:** **Google Gemini 1.5 Pro** synthesizes the answer based on the retrieved context.

## 🚀 Features
* **Drag-and-Drop UI:** Upload any PDF textbook via the Streamlit interface.
* **Context-Aware Chat:** Ask detailed questions about specific chapters or concepts.
* **Secure:** API keys are handled via session input, ensuring security.

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Chandrabhan-Choudhary/DocuMind-AI-Study-Assistant.git](https://github.com/Chandrabhan-Choudhary/DocuMind-AI-Study-Assistant.git)
    cd DocuMind-AI-Study-Assistant
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **Access the UI**
    Open your browser to `http://localhost:8501`

## 🔮 Future Roadmap
* [ ] Generate Quizzes from PDF content.
* [ ] Add "Summarize Chapter" button.
* [ ] Support for multiple document uploads.

---
*Built by [Chandrabhan Choudhary](https://www.linkedin.com/in/chandrabhan-choudhary/)*