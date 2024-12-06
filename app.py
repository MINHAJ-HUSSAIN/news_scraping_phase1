import requests
from newspaper import Article
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
import time
import shutil

# Streamlit app title and sidebar
st.title("News Scraper and QA System")
st.sidebar.title("News Article URLs")

# Input fields for URLs
url1 = st.sidebar.text_input("Enter URL 1:")
url2 = st.sidebar.text_input("Enter URL 2:")
process_urls = st.sidebar.button("Process URLs")
question = st.text_input("Ask a question:")

# Placeholder for feedback
status_placeholder = st.empty()

# Embedding and LLM configuration
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
Groq_API_TOKEN = os.getenv("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")
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=Groq_API_TOKEN,
    temperature=0.7,
    max_tokens=1024,
)

# Path to store vector database
vector_database_filepath = "vectorstore_dataset"

# Function to scrape articles
def newsscrapper(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            article = Article("")
            article.set_html(response.text)
            article.parse()
            cleaned_text = "\n".join(
                [line.strip() for line in article.text.split("\n") if line.strip()]
            )
            metadata = {
                "source": url,
                "Author": ", ".join(article.authors),
                "Publication date": str(article.publish_date),
                "Article Title": article.title,
            }
            return Document(page_content=cleaned_text, metadata=metadata)
        else:
            return None
    except Exception as e:
        return None

# Processing URLs
if process_urls:
    urls = [url1, url2]
    documents = [newsscrapper(url) for url in urls if url]
    documents = [doc for doc in documents if doc]
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=0
        )
        chunks = []
        for doc in documents:
            chunks.extend(text_splitter.split_documents([doc]))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vector_database_filepath)
        status_placeholder.text("Vector database created successfully.")
    else:
        status_placeholder.text("Failed to scrape URLs or no valid URLs provided.")

# Answering questions
if question:
    if os.path.exists(vector_database_filepath):
        vectorstore = FAISS.load_local(vector_database_filepath, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        response = chain.invoke({"question": question})
        st.subheader("Answer")
        st.write(response["answer"])
        st.subheader("Sources")
        st.write(response.get("sources", ""))
    else:
        st.write("No vector database found. Please process URLs first.")

