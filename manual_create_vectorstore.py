from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load your local document
loader = TextLoader("data/manual_test.txt")
documents = loader.load()

print(f"📄 Loaded {len(documents)} document(s)")

# Use HuggingFace embeddings (free, no API key required)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("🧠 Generating embeddings...")
vectorstore = FAISS.from_documents(documents, embeddings)

print("⚙️ Creating vectorstore...")
vectorstore.save_local("vectorstore")

print("✅ Vectorstore created successfully!")
