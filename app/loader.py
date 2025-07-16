import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(folder_path: str):
    docs = []
    print(f"📂 Loading from folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(f"🔍 Checking: {filename}")

        if filename.endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(file_path)
                loaded = loader.load()
                docs.extend(loaded)
                print(f"✅ Loaded {len(loaded)} pages from {filename}")
            except Exception as e:
                print(f"❌ Failed to load PDF {filename}: {e}")

        elif filename.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                loaded = loader.load()
                docs.extend(loaded)
                print(f"✅ Loaded text from {filename}")
            except Exception as e:
                print(f"❌ Failed to load TXT {filename}: {e}")

    print(f"📄 Total documents loaded: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Total chunks created: {len(chunks)}")
    return chunks
