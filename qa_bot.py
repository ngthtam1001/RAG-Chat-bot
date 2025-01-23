from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def create_chain(vector_db_path):
    # Load embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Set up HuggingFace pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create prompt
    prompt = PromptTemplate(
        template="Context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    
    # Create chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs={'prompt': prompt}
    )

chain = create_chain(r"C:/Users/admin/Desktop/SimpleRAG/vectordb")
response = chain.invoke({"query": "What is principle 7 of B.Control Environment of year 2022?"})
print(response['result'])