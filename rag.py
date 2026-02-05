# rag.py

import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#from langchain_community.vectorstores import Chroma
#from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import cohere


from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)

print("Using Pinecone index:", os.environ.get("PINECONE_INDEX_NAME"))


# ---------------------------
# 1. Load documents
# ---------------------------
def load_documents(file_path: str) -> List[Document]:
    """
    Load a document from disk based on file type.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    else:
        raise ValueError("Unsupported file format")


# ---------------------------
# 2. Split documents
# ---------------------------
def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for semantic retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


# ---------------------------
# 3. Embeddings model
# ---------------------------
def load_embedding_model():
    """
    Load the sentence embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------
# 4. Vector store
# ---------------------------
def build_vector_store(chunks, embeddings):
    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )




def load_vector_store(embeddings):
    return PineconeVectorStore.from_existing_index(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings,
    )




# ---------------------------
# 5. Retrieval
# ---------------------------
def retrieve_chunks(query: str, vector_store) -> List[Document]:
    """
    Retrieve top-k relevant chunks for a query.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    return retriever.invoke(query)

#---------------------------
# 5.5 Cohere Retrieval
#---------------------------
def rerank_chunks(query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    """
    Rerank retrieved chunks using Cohere Rerank.
    """
    if not docs:
        return docs

    co = cohere.Client(os.environ["COHERE_API_KEY"])

    texts = [doc.page_content for doc in docs]

    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=min(top_n, len(texts)),
    )

    reranked_docs = [docs[result.index] for result in rerank_results.results]
    return reranked_docs



# ---------------------------
# 6. Generation
# ---------------------------
def generate_answer(query: str, retrieved_docs: List[Document]) -> str:
    numbered_context = []

    for i, doc in enumerate(retrieved_docs, start=1):
        numbered_context.append(
            f"[{i}] {doc.page_content}"
        )

    context = "\n\n".join(numbered_context)

    prompt = f"""
You are answering questions using retrieved document excerpts.

Rules:
- Use ONLY the provided context.
- Cite statements using square brackets like [1], [2].
- If the answer is not present in the context, say so clearly.
- Do NOT use prior knowledge.

Context:
{context}

Question:
{query}

Answer clearly and concisely with citations.
"""

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    return llm.invoke(prompt).content

