"""
PDF Vectorization Pipeline - Retrieval Endpoint
FastAPI server exposing POST /query
No LLM chain — raw similarity_search_with_score() only.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Config ───────────────────────────────────────────────────────────────────────

PERSIST_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── App bootstrap ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF Vector Retrieval API",
    description="Similarity search over a LangChain/Chroma vector store. No LLM chains.",
    version="1.0.0",
)

# Load once at startup
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


@app.on_event("startup")
def load_vectorstore():
    global _embeddings, _vectorstore
    print(f"[startup] Loading embedding model: {EMBEDDING_MODEL}")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[startup] Connecting to Chroma at: {PERSIST_DIR}")
    _vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_embeddings,
        collection_name=COLLECTION_NAME,
    )
    count = _vectorstore._collection.count()
    print(f"[startup] Vector store ready. {count} vector(s) available.")


# ── Request / Response models ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language query string.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of top results to return.")


class ChunkResult(BaseModel):
    chunk_text: str
    page_number: Optional[int]
    score: float


class QueryResponse(BaseModel):
    results: List[ChunkResult]


# ── Endpoint ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest) -> QueryResponse:
    """
    Embed the query with the same model used during ingestion,
    then call similarity_search_with_score() directly on the vector store.
    No RetrievalQA, no LLM chain.
    """
    if _vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised.")

    raw_results = _vectorstore.similarity_search_with_score(
        query=body.query,
        k=body.top_k,
    )

    results: List[ChunkResult] = []
    for doc, score in raw_results:
        page_number = doc.metadata.get("page")
        if page_number is not None:
            # PyPDFLoader uses 0-based pages; convert to 1-based for humans
            page_number = int(page_number) + 1
        results.append(
            ChunkResult(
                chunk_text=doc.page_content,
                page_number=page_number,
                score=float(score),
            )
        )

    return QueryResponse(results=results)


@app.get("/health")
def health():
    count = _vectorstore._collection.count() if _vectorstore else 0
    return {"status": "ok", "vectors_in_store": count}


# ── Entry-point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
