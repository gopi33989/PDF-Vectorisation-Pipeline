"""
PDF Vectorization Pipeline - Ingestion Script
LangChain + LangGraph Implementation
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# LangGraph imports
from langgraph.graph import StateGraph, END

# ── State schema ────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    file_path: str
    documents: List[Document]
    chunks: List[Document]
    vectorstore_path: str
    status: str


# ── Node functions ───────────────────────────────────────────────────────────────

def load_node(state: PipelineState) -> PipelineState:
    """Node 1 – LOAD: Load the PDF using PyPDFLoader."""
    print(f"\n[LOAD] Loading PDF: {state['file_path']}")
    loader = PyPDFLoader(state["file_path"])
    documents = loader.load()
    print(f"[LOAD] Loaded {len(documents)} page(s).")
    return {**state, "documents": documents, "status": "loaded"}


def split_node(state: PipelineState) -> PipelineState:
    """Node 2 – SPLIT: Chunk documents with RecursiveCharacterTextSplitter."""
    print(f"\n[SPLIT] Splitting {len(state['documents'])} document(s) …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(state["documents"])
    print(f"[SPLIT] Produced {len(chunks)} chunk(s).")
    return {**state, "chunks": chunks, "status": "split"}


def embed_node(state: PipelineState) -> PipelineState:
    """Node 3 – EMBED: Generate embeddings (model loaded here; persisted in store_node)."""
    print("\n[EMBED] Loading embedding model (all-MiniLM-L6-v2) …")
    # Instantiation triggers the model download on first run
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[EMBED] Embedding model ready.")
    # Attach to state so store_node can reuse the same instance
    return {**state, "embeddings": embeddings, "status": "embedded"}


def store_node(state: PipelineState) -> PipelineState:
    """Node 4 – STORE: Persist embeddings to a Chroma vector store."""
    print(f"\n[STORE] Persisting {len(state['chunks'])} chunk(s) to Chroma …")
    persist_dir = state.get("vectorstore_path", "./chroma_db")

    vectorstore = Chroma.from_documents(
        documents=state["chunks"],
        embedding=state["embeddings"],
        persist_directory=persist_dir,
        collection_name="pdf_chunks",
    )
    print(f"[STORE] Vector store saved to: {persist_dir}")
    count = vectorstore._collection.count()
    print(f"[STORE] Total vectors stored: {count}")
    return {**state, "status": "stored"}


# ── Build LangGraph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("load",  load_node)
    graph.add_node("split", split_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("load")
    graph.add_edge("load",  "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)

    return graph.compile()


# ── CLI entry-point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into a Chroma vector store via LangGraph."
    )
    parser.add_argument("--file", required=True, help="Path to the PDF file.")
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Directory to persist the Chroma vector store (default: ./chroma_db).",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    pipeline = build_graph()

    # Print node list for README / evaluation
    print("\n── LangGraph Node List ─────────────────────────────")
    print("  load  →  split  →  embed  →  store  →  END")
    print("────────────────────────────────────────────────────\n")

    initial_state: PipelineState = {
        "file_path": args.file,
        "documents": [],
        "chunks": [],
        "vectorstore_path": args.persist_dir,
        "status": "init",
    }

    final_state = pipeline.invoke(initial_state)
    print(f"\n✅ Ingestion complete. Status: {final_state['status']}")


if __name__ == "__main__":
    main()
