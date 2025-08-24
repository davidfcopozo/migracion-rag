# PDF RAG (Ley de Migración)

## Purpose

Concise technical reference for the Streamlit RAG app in this repo. The app answers questions about the Ley de Migración de la República Dominicana by ingesting a PDF, creating vector embeddings, and using an LLM to produce structured answers.

## Key files

- `app.py` — main Streamlit application and orchestration.
- `requirements.txt` — Python dependencies used for testing/development.
- `data/` — place your PDF(s) here (default expected: `LEY-DE-MIGRACION.pdf`).
- `chat_history.json` — persisted chat history (created at runtime).

## How it works (high level)

1. Discover a PDF under `./data/` (prefers `LEY-DE-MIGRACION.pdf`).
2. Load the PDF with `UnstructuredPDFLoader` and split text with `RecursiveCharacterTextSplitter`.
3. Produce embeddings for each chunk using one of:
   - Local: `OllamaEmbeddings` (`nomic-embed-text`) in local mode.
   - Cloud: `HuggingFaceEmbeddings` (preferred), or anonymous HF access, or a local `sentence-transformers` fallback.
4. Store embeddings in a vector store:
   - Local: Chroma (persisted under `./local_dbs/`).
   - Cloud: Pinecone (index name controlled by `PINECONE_INDEX_NAME`).
5. Create a `MultiQueryRetriever` from the vector store and construct a prompt chain. The chain is invoked with `{"context": ..., "question": ...}`.
6. Stream the LLM response progressively in the Streamlit chat using conservative formatting, then render final Markdown.

## Running locally (development)

1. Activate your venv and install deps:

```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the app:

```powershell
streamlit run app.py
```

3. Ensure a PDF is present under `./data/` or edit `DOC_PATH` in `app.py`.

## Cloud mode (Groq + Pinecone + HF)

Set environment variables in the shell _before_ starting Streamlit so worker processes inherit them:

- `DEPLOYMENT_MODE=cloud`
- `GROQ_API_KEY` (Groq provider)
- `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` (Pinecone)
- `PINECONE_INDEX_NAME` (optional; defaults to `rag-index`)
- `HUGGINGFACE_API_TOKEN` (preferred; optional anonymous fallback attempted)

Example (PowerShell):

```powershell
$env:DEPLOYMENT_MODE = "cloud";
$env:PINECONE_API_KEY = "<key>";
$env:HUGGINGFACE_API_TOKEN = "<token>";
streamlit run app.py
```

Notes:

- The app attempts to detect embedding dimension at runtime and will create or choose a Pinecone index that matches that dimension (it may create `PINECONE_INDEX_NAME-<dim>` if needed).
- If the Pinecone index is empty, the app will automatically ingest the PDF and populate it on first run.

## Important environment variables

- `DEPLOYMENT_MODE` — `local` (default) or `cloud`.
- `GROQ_API_KEY` — required in cloud mode.
- `HUGGINGFACE_API_TOKEN` — used by `HuggingFaceEmbeddings`; the app also tries anonymous access or a local fallback.
- `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME` — Pinecone configuration.

## Embedding fallbacks and behavior

- The app first tries `HuggingFaceEmbeddings(model_name=...)` with provided HF token.
- If that fails it tries anonymous HF for public models.
- As a final fallback it attempts to load a local model with `sentence-transformers` or `transformers` + `torch`.
- Embedding vector dimension is probed at runtime using `embed_query("dim-check")` and used to select/create a compatible Pinecone index.

## Streaming & formatting

- Progressive streaming uses `format_response_progressive()` to avoid markdown headers that change text size during streaming.
- Final output is rendered with `format_response_md()` which adds conservative markdown and a footer referencing the law.

## Troubleshooting

- HF 401 in Streamlit: ensure HF token is exported in the same shell you start Streamlit; Streamlit worker subprocesses inherit env vars only from the launching process.
- Pinecone index dimension mismatch: the app automatically selects/creates an index suffixed with the embedding dimension if a mismatch is detected.
- If Pinecone remains empty after first run: check Streamlit console logs for the first error during index creation or upsert; common causes are invalid API key, region mismatch, or insufficient permissions.

## Notes & future improvements

- Add an explicit "reindex" button to force re-creation of the vector index.
- Add a toggle in the UI to pick embedding backend manually for easier debugging.
- Consider persisting more diagnostics (index_name used, embed backend, embed_dim) to `chat_history.json` or a small `diagnostics.log` file.

---

Generated from the development session that added embedding fallbacks, runtime dimension handling, and Pinecone auto-population logic.
