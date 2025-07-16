import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from .vectorstore import load_vectorstore

# Load .env variables
load_dotenv()

# Use Ollama's mistral model
llm = ChatOllama(model="mistral")

def get_prompt_template():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant and an expert in Microsoft Admin Center.

Only answer questions that are related to Microsoft Admin Center using the context provided below.

If the question is unrelated to Microsoft Admin Center (like sports, weather, movies, food, etc.), respond with:
"I'm only trained to answer questions related to Microsoft Admin Center."

---------------------
Context:
{context}

Question:
{question}

Answer:
"""
    )

def get_qa_chain() -> RetrievalQA:
    # Use a local embedding model from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load your local vectorstore
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Get the prompt
    prompt = get_prompt_template()

    # Set up the QA chain with Mistral
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain
