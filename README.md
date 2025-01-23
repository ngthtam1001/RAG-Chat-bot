# SimpleRAG Documentation

## Overview
SimpleRAG is a Python-based implementation designed to facilitate information retrieval and response generation using language models and vector databases. It employs HuggingFace models for language generation and embeddings, integrating them with LangChain utilities for advanced query handling.

---

## Features
1. Language Model Loading:
   - Uses HuggingFace models like openai-community/gpt2 for language generation.
   - Embedding models: sentence-transformers/all-MiniLM-L6-v2.

2. Vector Database:
   - Implements FAISS (Facebook AI Similarity Search) for efficient vector-based information retrieval.
   - Supports loading documents (e.g., PDFs) and splitting them into manageable text chunks.

3. Query-Answer Chain:
   - Builds a query-answer pipeline with prompt templates and text generation.
   - Supports context-aware question answering using a retrieval-based approach.

---

## File Descriptions

### 1. load_model.py
- Loads and saves HuggingFace models (`GPT-2` and `MiniLM`).
- Stores the pre-trained models locally for faster access.

### 2. qa_bot.py
- Implements a question-answering system using LangChain’s RetrievalQA.
- Loads vector databases and creates an LLM pipeline for query handling.
- Example query provided in the script for demonstration.

### 3. simple_chain.py
- Demonstrates a simplified query-answering chain.
- Includes functions to load models, define prompt templates, and generate responses.
- Focuses on minimal input-output functionality.

### 4. vector_db.py
- Manages document processing and vector database creation.
- Supports loading PDF files and splitting content into chunks for embedding generation.
- Saves the FAISS vector database for subsequent retrieval.

---

## Prerequisites
1. Hardware:
   - GPU recommended for faster processing.

2. Software:
   - Python 3.8+
   - Libraries:
     - torch
     - transformers
     - langchain
     - sentence-transformers
     - faiss-cpu (or `faiss-gpu`)

3. Environment Variables:
   - TF_ENABLE_ONEDNN_OPTS set to 0 for model compatibility.

---

## Usage

### Setting Up Models
1. Run load_model.py to download and save the required models locally.
2. Ensure the model paths in the script match your local setup.

### Creating a Vector Database
1. Place your PDF files in the data directory.
2. Update the data_path and vector_db_path in vector_db.py.
3. Execute vector_db.py to create and save the FAISS database.

### Running the QA System
1. Ensure the vector database path matches in qa_bot.py.
2. Use the create_chain function to initialize the retrieval-based query-answering system.
3. Pass queries to the system for contextual responses.

### Simple Query Chain
1. Customize prompts and questions in simple_chain.py.
2. Execute the script for quick answers without vector database integration.

---

## Examples

### Question-Answer Example
- Input: "What is principle 7 of B.Control Environment of year 2022?"
- Output: Contextual answer generated from documents.

### Simple Query Example
- Input: "How many hours a day?"
- Output: "Depends on your schedule." (or similar output).

---

## Troubleshooting
- CUDA Not Found: Ensure GPU drivers and CUDA are installed properly.
- Model Path Errors: Verify that the paths in the scripts match your local setup.
- Missing Libraries: Install required libraries using pip install.
