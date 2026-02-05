# 📄 RAG Document Q&A System (Predusk AI Assessment)

This project implements a **production-oriented Retrieval-Augmented Generation (RAG)** system that allows users to upload documents and ask questions grounded strictly in the document content.  
The system uses **cloud-hosted vector storage**, **semantic retrieval**, **reranking**, and **inline citations** to ensure accurate, explainable, and hallucination-safe answers.

---

## 🚀 Live Demo
**Live App URL:**  
 *( deployed Streamlit )*

---

## 📂 GitHub Repository
This repository contains the complete source code, setup instructions, and documentation for the Predusk AI technical assessment.

---

## 🧠 Architecture Overview

### High-Level Flow

Document Upload
↓
Chunking
↓
Embedding Generation
↓
Pinecone Vector Database (ANN Search)
↓
Top-K Retrieval
↓
Cohere Reranker
↓
LLM Answer Generation
↓
Answer with Inline Citations + Source Chunks


---

## 🏗️ System Architecture

### Frontend
- Streamlit web interface
- Document upload (PDF / TXT)
- Question input
- Answer output with inline citations
- Expandable source context viewer

### Backend
- Python 3.11
- LangChain orchestration
- HuggingFace sentence embeddings
- Pinecone (cloud-hosted vector database)
- Cohere Rerank
- OpenAI LLM for answer generation

---

##  Key Design Decisions

### 1️ Retrieval-Augmented Generation (RAG)
Instead of sending entire documents to the LLM, the system retrieves only the most relevant document chunks and passes them as context.  
This improves:
- Answer accuracy
- Cost efficiency
- Latency
- Explainability

---

### 2️ Cloud-Hosted Vector Database
- **Pinecone (Serverless, AWS)**
- Dense embeddings
- Cosine similarity
- Approximate Nearest Neighbor (ANN) search

This satisfies scalability and cloud-deployment requirements.

---

### 3️ Chunking Strategy
- Chunk size: **800 tokens**
- Chunk overlap: **120 tokens**
- Recursive splitting preserves semantic coherence
- Designed for long technical documents (e.g., RFCs, standards)

---

### 4️ Reranking
Initial similarity search retrieves top-K chunks, followed by **Cohere Rerank** to:
- Improve relevance
- Handle structured content (tables, headers)
- Reduce noise before generation

---

### 5️ Inline Citations
Each retrieved chunk is assigned an index `[1]`, `[2]`, `[3]`.

The LLM is instructed to:
- Cite facts using these indices
- Use only provided context
- Explicitly state when information is not present

Source chunks are displayed below each answer.

---

##  Evaluation

### Test Dataset
- RFC 7231 — HTTP/1.1 Semantics and Content

### Sample Questions
- Which HTTP methods are defined in RFC 7231?
- What is the difference between safe and idempotent methods?
- Is POST idempotent?
- Does RFC 7231 define authentication mechanisms?

### Observations
- Accurate answers when relevant sections are retrieved
- Graceful failure when information is absent
- Inline citations correctly map to source chunks

---

##  Limitations & Future Improvements

- Table content split across chunks could benefit from:
  - Structure-aware table parsing
  - Hierarchical parent–child retrieval
- Hybrid retrieval (BM25 + embeddings)
- Query re-writing for improved recall
- Enhanced UI highlighting for citations

---

## 🛠️ Setup Instructions

### Prerequisites
- Python **3.11**
- Pinecone account
- Cohere account
- OpenAI API key

---

### 1️ Clone Repository
```bash
git clone https://github.com/prajithJ/RAG_CLOUD
cd RAG_CLOUD

### Create Virtual Environment
python -m venv rag
rag\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Create a .env file in the project root:
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=rag-doc
COHERE_API_KEY=your_cohere_key

### Launch the app using streamlit
streamlit run app.py

```

## Resume Link:
https://drive.google.com/file/d/1CLlgTpcRpEHd-xgXeRXFhKuQLt4TfEVb/view?usp=sharing



