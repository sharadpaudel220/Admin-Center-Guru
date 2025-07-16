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

# ✅ Use DeepSeek R1 running via Ollama
# e.g. deepseek-coder:6.7b or latest
llm = ChatOllama(model="deepseek-coder:6.7b")

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = get_prompt_template()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain
