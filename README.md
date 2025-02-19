## MediBot 🤖 - A Generative AI-Powered Medical Chatbot

## 📌 Overview
MediBot is an intelligent Generative AI-powered medical chatbot that provides users with medical guidance based on their queries. It leverages Natural Language Processing (NLP), Pinecone for vector search, and LLaMA 2 to deliver accurate and relevant responses.

## 🚀 Tech Stack:

Generative AI (LLaMA 2)

Pinecone Vector Database

Flask (Backend)

Hugging Face Transformers

LangChain for RetrievalQA

Sentence Transformers

## ⚙️ Features
✅ Medical Query Processing – Provides AI-driven medical responses

✅ Pinecone-Powered Search – Uses vector embeddings for fast retrieval

✅ Hugging Face Integration – Implements state-of-the-art NLP models

✅ Secure API Communication – Ensures privacy in medical conversations

✅ Web Interface – Easy-to-use chat UI


## 🚀 Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/Kamleshkumar01/MediBot.git

cd MediBot

2️⃣ Set Up a Virtual Environment (Optional but Recommended)

python -m venv env

source env/bin/activate   # For Linux/macOS

env\Scripts\activate      # For Windows


3️⃣ Install Dependencies

pip install -r requirements.txt


4️⃣ Set Up Environment Variables

Create a .env file in the root directory and add your API keys:

PINECONE_API_KEY=your_pinecone_api_key

PINECONE_ENV=your_pinecone_env



5️⃣ Run the Flask App

python app.py

🎉 Your chatbot is now running! Open http://127.0.0.1:8080 in your browser.


# 🛠️ Technologies Used
🔹 Python (Flask, LangChain, Hugging Face)
🔹 Pinecone (Vector Search for AI-driven retrieval)
🔹 LLaMA 2 (Generative AI Model)
🔹 HTML, CSS, JavaScript (Frontend)
