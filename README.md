# RAG Chatbot
Simple RAG chatbot using GPT-2 and MiniLM models.
## Features:

Document ingestion and vectorization
Semantic search using FAISS
Question answering using GPT-2
Embeddings using MiniLM

## Usage:

Install dependencies: pip install -r requirements.txt
Load models: python code/load_model.py
Create vector database: python code/vector_db.py
Run QA bot: python code/qa_bot.py

## Models:

LLM: openai-community/gpt2
Embeddings: sentence-transformers/all-MiniLM-L6-v2
