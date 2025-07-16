import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Constants
VECTOR_DIR = "vectorstore"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(documents):
    if not documents:
        raise ValueError("❌ No documents to index. Aborting.")
    print("⚙️ Creating vectorstore using HuggingFace embeddings...")
    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
    vectorstore.save_local(VECTOR_DIR)
    print("✅ Vectorstore created and saved to:", VECTOR_DIR)

def load_vectorstore():
    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ Vectorstore not found at {index_path}")
    print("📦 Loading existing vectorstore...")
    return FAISS.load_local(VECTOR_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
