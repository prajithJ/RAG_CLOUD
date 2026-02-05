# app.py

import os
from dotenv import load_dotenv
from pathlib import Path


# ---- Load ENV FIRST ----
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

import streamlit as st

import shutil
from pathlib import Path

from rag import (
    load_documents,
    split_documents,
    load_embedding_model,
    build_vector_store,
    load_vector_store,
    retrieve_chunks,
    generate_answer,
    rerank_chunks
)

#from config import VECTOR_DB_DIR

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("📄 RAG Document Q&A")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment.")
    st.stop()


# ---- Sidebar ----
st.sidebar.header("Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or TXT file", type=["pdf", "txt"]
)

process_button = st.sidebar.button("Process Document")

# ---- Load models once ----
@st.cache_resource
def load_models():
    return load_embedding_model()

embeddings = load_models()

# ---- Process document ----
if process_button and uploaded_file:
    with st.spinner("Processing document..."):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_documents(temp_path)
        chunks = split_documents(docs)
        build_vector_store(chunks, embeddings)

        os.remove(temp_path)

        st.sidebar.success("Document processed and indexed.")

# ---- Load vector DB if exists ----
vector_store = load_vector_store(embeddings)

# ---- Q&A ----
st.header("Ask a Question")

query = st.text_input("Enter your question")

if st.button("Get Answer") and query:
    if not vector_store:
        st.warning("Please upload and process a document first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            retrieved_docs = retrieve_chunks(query, vector_store)
            reranked_docs = rerank_chunks(query, retrieved_docs)
            answer = generate_answer(query, reranked_docs)

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Sources"):
                for i, doc in enumerate(reranked_docs, start=1):
                    st.markdown(f"**[{i}] Source Chunk**")
                    st.write(doc.page_content)
