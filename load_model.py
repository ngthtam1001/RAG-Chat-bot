import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_models():
    llm_path = "C:/Users/admin/Desktop/SimpleRAG/model/OpenAI"
    embedding_path = "C:/Users/admin/Desktop/SimpleRAG/model/MiniLM"
    
    # Load LLM
    llm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    llm_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    llm.save_pretrained(llm_path)
    llm_tokenizer.save_pretrained(llm_path)
    
    # Load embeddings
    embeddings = AutoModelForCausalLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") 
    embeddings_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embeddings.save_pretrained(embedding_path)
    embeddings_tokenizer.save_pretrained(embedding_path)
    
load_models()