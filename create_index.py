from app.loader import load_documents, split_documents
from app.vectorstore import create_vectorstore

def main():
    docs = load_documents(r"C:\Users\SHARAD.IT-IFC-L1\Documents\Admin-Center-Guru\data")
    print(f"✅ Documents loaded: {len(docs)}")

    chunks = split_documents(docs)
    print(f"✅ Chunks created: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ No chunks to index! Check your files — they might be empty or unreadable.")
        return

    create_vectorstore(chunks)
    print("✅ Vectorstore successfully created!")

if __name__ == "__main__":
    main()
