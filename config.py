# config.py

# ---- Models ----
LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- Chunking ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---- Retrieval ----
TOP_K = 6

# ---- Storage ----
#VECTOR_DB_DIR = "chroma_db"
PINECONE_INDEX_NAME = "rag-doc"

