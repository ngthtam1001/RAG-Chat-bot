from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

def create_db(data_path: str, vector_db_path: str):
    loader = DirectoryLoader(data_path, glob = "*.pdf", loader_cls= PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 50)
    chunks = text_splitter.split_documents(documents)

    embedding_model  = GPT4AllEmbeddings(model_file =  r"C:\Users\admin\Desktop\SimpleRAG\model\MiniLM\model.safetensors", device = 'cuda')
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

data_path = r"C:\Users\admin\Desktop\SimpleRAG\data"
vector_db_path = r"C:\Users\admin\Desktop\SimpleRAG\vectordb"
create_db(data_path= data_path, vector_db_path=vector_db_path)