# PDF Vectorization Pipeline — LangChain / LangGraph

A two-part pipeline that ingests a PDF into a Chroma vector store and exposes a raw similarity-search HTTP endpoint. No LLM chains are used on the retrieval side.

---

## Live Deployment

> **API base URL:** `https://<your-deployment-url>`  
> (Replace with your Railway / Render / Fly.io URL after deployment)

---

## Project Structure

```
.
├── ingest.py          # LangGraph ingestion pipeline
├── server.py          # FastAPI retrieval server
├── requirements.txt
├── results.json       # Sample query outputs
└── README.md
```

---

## Setup & Installation

### 1 — Clone the repo

```bash
git clone git@github.com:<you>/yourName_Nestack_Submission.git
cd yourName_Nestack_Submission
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. No API key required.

---

## Part 1 — Ingestion

### Run

```bash
python ingest.py --file document.pdf
```

Optional: override the vector store directory:

```bash
python ingest.py --file document.pdf --persist-dir ./my_chroma_db
```

### Sample output

```
── LangGraph Node List ─────────────────────────────
  load  →  split  →  embed  →  store  →  END
────────────────────────────────────────────────────

[LOAD] Loading PDF: document.pdf
[LOAD] Loaded 2 page(s).

[SPLIT] Splitting 2 document(s) …
[SPLIT] Produced 14 chunk(s).

[EMBED] Loading embedding model (all-MiniLM-L6-v2) …
[EMBED] Embedding model ready.

[STORE] Persisting 14 chunk(s) to Chroma …
[STORE] Vector store saved to: ./chroma_db
[STORE] Total vectors stored: 14

✅ Ingestion complete. Status: stored
```

---

## LangGraph — Ingestion Graph

The ingestion is modelled as a directed acyclic graph with four named nodes:

```
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│  load  │────▶│ split  │────▶│ embed  │────▶│ store  │────▶ END
└────────┘     └────────┘     └────────┘     └────────┘
```

| Node    | Responsibility                                      |
|---------|-----------------------------------------------------|
| `load`  | `PyPDFLoader` — reads PDF pages into `Document` objects |
| `split` | `RecursiveCharacterTextSplitter` — chunks each page |
| `embed` | Loads `HuggingFaceEmbeddings` model into state      |
| `store` | `Chroma.from_documents()` — persists to disk        |

Each node receives the full pipeline state and returns an updated copy, keeping side effects isolated and the graph inspectable.

---

## Part 2 — Retrieval Server

### Run

```bash
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Endpoint

#### `POST /query`

**Request body**

```json
{
  "query": "What are the ingestion pipeline steps?",
  "top_k": 3
}
```

**Response**

```json
{
  "results": [
    {
      "chunk_text": "Write a script using LangChain primitives ...",
      "page_number": 1,
      "score": 0.2814
    }
  ]
}
```

> **Score** is the Chroma L2 distance — lower means more similar.

#### `GET /health`

```json
{ "status": "ok", "vectors_in_store": 14 }
```

### Example curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the evaluation criteria?", "top_k": 3}'
```

---

## Component Choice Justification

### Document Loader — `PyPDFLoader`

`PyPDFLoader` is the lightest-weight LangChain PDF loader, has zero external service dependencies, and preserves per-page metadata (the `page` key) out of the box. This metadata is surfaced directly in the `/query` response as `page_number`. For a structured, text-heavy assessment PDF, there is no need for `UnstructuredPDFLoader`'s heavier layout analysis.

### Text Splitter — `RecursiveCharacterTextSplitter`

`RecursiveCharacterTextSplitter` tries progressively smaller separators (`\n\n`, `\n`, `.`, ` `) before hard-cutting on character count, so chunks respect paragraph and sentence boundaries wherever possible.

**Chunk size — 500 characters**  
The assessment PDF contains short, dense bullet points and table rows. At 500 characters a chunk captures one to three coherent bullet points without merging semantically unrelated sections. Larger chunks (e.g. 1 000) would mix evaluation criteria with submission instructions in the same vector, hurting retrieval precision.

**Overlap — 100 characters (~20 %)**  
A 20 % overlap means the tail of a bullet point is repeated at the head of the next chunk. This ensures queries whose answer spans a sentence boundary will still match at least one chunk confidently, without the redundancy penalty of a 50 % overlap.

### Embedding Model — `sentence-transformers/all-MiniLM-L6-v2`

- Runs fully locally — no API key, no network call at query time.
- 384-dimensional embeddings are fast to compute on CPU and compact in Chroma.
- Trained on 1 B+ sentence pairs; strong semantic similarity on English technical text.
- A drop-in HuggingFace integration means the same `HuggingFaceEmbeddings` instance is used identically in both `ingest.py` and `server.py`, guaranteeing embedding-space consistency.

### Vector Store — `Chroma`

- Native LangChain integration with `from_documents()` and `similarity_search_with_score()`.
- Disk persistence with a single `persist_directory` argument — no external service needed.
- `similarity_search_with_score()` returns raw `(Document, float)` tuples, satisfying the constraint of no RAG chain wrapping.
- `collection_name` parameter keeps multiple ingestion runs isolated.

---

## Sample Query Results

See [`results.json`](./results.json) for four queries run against the assessment PDF, each returning the top-3 most relevant chunks with `chunk_text`, `page_number`, and `score`.

---

## Deployment (Railway / Render / Fly.io)

1. Push the repo (private).
2. Set start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
3. Either commit `chroma_db/` or run `ingest.py` as a one-off job before starting the server.
4. Paste the live URL in this README and in your submission.

---

## Constraints Checklist

- [x] Python only
- [x] `PyPDFLoader` for loading
- [x] `RecursiveCharacterTextSplitter` for splitting
- [x] `HuggingFaceEmbeddings` for embeddings
- [x] `Chroma` with built-in persistence for the vector store
- [x] `/query` calls `similarity_search_with_score()` directly — no `RetrievalQA`, no LLM chain
- [x] LangGraph with named nodes (`load → split → embed → store`)
- [x] `results.json` with ≥ 3 queries
