# Architecture — PDF RAG App (Ley de Migración)

## Overview

This document describes the high-level architecture and data flow for the Streamlit-based RAG application that answers questions about the Dominican Republic migration law.

## Components

- UI: Streamlit chat UI with progressive streaming and final Markdown rendering.
- Document loader: `UnstructuredPDFLoader` (loads PDF pages into Document objects).
- Splitter: `RecursiveCharacterTextSplitter` (chunks long text into overlapping passages).
- Embeddings:
  - Local: `OllamaEmbeddings` (nomic embedding) for local deployment.
  - Cloud: `HuggingFaceEmbeddings` (primary), anonymous HF fallback, or local `sentence-transformers` if HF unavailable.
- Vector store:
  - Local: Chroma
  - Cloud: Pinecone (index selection/creation based on embedding dimension)
- Retriever: `MultiQueryRetriever` (generates multiple query rewrites using the LLM to improve recall).
- LLMs:
  - Local: Ollama (`ChatOllama`) for on-device inference.
  - Cloud: Groq (`ChatGroq`) for managed inference.

## Data flow

1. PDF discovery: app finds `./data/LEY-DE-MIGRACION.pdf` or the first PDF in `./data/`.
2. Load & chunk: the PDF is loaded and split into chunks suitable for embedding.
3. Embed & store: chunks are converted to vectors and stored in Chroma (local) or Pinecone (cloud).
4. Query handling:
   - User question arrives via the Streamlit chat.
   - `MultiQueryRetriever` generates alternative queries using the LLM and retrieves top documents from the vector store.
   - Context is assembled from retrieved chunks and passed to the LLM chain.
   - The LLM returns a structured answer; the app streams it into the chat progressively then renders a final Markdown version.

## Operational details

- Embedding dimension compatibility: the app probes the embedding vector size and chooses/creates a Pinecone index with a matching dimension (`<index-name>-<dim>` if needed).
- Auto-population: when a Pinecone index is detected as empty (stats unavailable or vector count = 0), the app attempts to ingest the local PDF and populate the index automatically.
- Fallbacks: HF auth failures are handled by trying anonymous access; if that fails the app falls back to a local embedding model.
- Persistence: chat history is saved to `chat_history.json` for continuity across sessions.

## Security and secrets

- Keep API keys out of source control. Use environment variables or host secret management when deploying.
- Start Streamlit from a shell that has the environment variables set so subprocesses see required tokens.

## Deployment notes

- Local demo: run on a development machine with Ollama and Chroma available.
- Cloud demo: requires valid `GROQ_API_KEY`, `PINECONE_API_KEY`, and `HUGGINGFACE_API_TOKEN`. Verify Pinecone project/region and that API key has index create/upsert permissions.

## Diagram (conceptual)

User (browser)
↕
Streamlit UI (chat + streaming)
↕
Retriever + Chain (MultiQueryRetriever -> LLM chain)
↕
Vector Store <---- Embeddings <- Chunker <- PDF Loader
(Pinecone/Chroma) (HF / local / Ollama)

## Suggested next steps

- Add an admin UI to force reindexing and show current index name/embed_dim used.
- Export diagnostic logs for embedding backend and Pinecone operations.
- Add unit tests for ingestion, embedding dimension detection, and index selection logic.

---

Keep this doc updated as the project evolves. It was created to reflect the recent changes that added embedding fallbacks, runtime dimension handling, and improved Pinecone initialization logic.
