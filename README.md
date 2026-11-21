# Visual RAG Studio

Visual RAG Studio is a local-first visual RAG workflow builder. The current milestone includes **Story 3: Visual Builder UI**: a React Flow editor with a backend-driven node palette, schema-based Inspector, and real backend graph validation. It does not yet ingest documents or run RAG workflows.

The Architecture Gallery retains all 37 requested architecture families and patterns in [the capability registry](docs/capability-registry.md). Their catalog presence does not mean a runnable implementation exists. Traditional RAG is the first deterministic execution template; the gallery keeps it conservatively marked planned until a live Ollama integration run is verified. Naive RAG remains a separately defined, simpler direct top-k retrieve-to-generate pattern. See the [product requirements](docs/PRD.md), [architecture](docs/architecture.md), and [M1 stories](docs/epics/README.md).

## Requirements

- Node.js 20.19+ or 22.12+ (verified locally with Node 24.21.0).
- Python 3.11–3.14 (verified locally with Python 3.14.8). Use a supported regular GIL build.
- npm and uv (verified locally: npm 11.19.0, uv 0.12.24).
- Sentence Transformers provides default local multilingual embeddings. Generation is a separate provider: Ollama or Hugging Face Inference Providers.

## Local setup

From the repository root in PowerShell:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
cd backend
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --python 3.14 --group dev
cd ..
cd frontend
npm install
cd ..
```

The backend lock is `backend/uv.lock`; `frontend/package-lock.json` locks frontend dependencies. Package installation needs registry access.

On Windows, if `uv run` reports that `.venv\Scripts\python.exe` is blocked or points to an unavailable managed interpreter, recreate the generated environment using the Python installation you have:

```powershell
cd backend
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --clear --python "C:\Path\To\Python\python.exe" .venv
uv sync --python "C:\Path\To\Python\python.exe" --group dev
```

Replace that example path with the actual installed Python executable. `.venv` and `.uv-cache` are generated and ignored.

Start the backend in one terminal:

```powershell
cd backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Start the frontend in another:

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite (normally http://127.0.0.1:5173). If that port is already occupied, Vite automatically selects the next available port. Vite proxies `/api` to the loopback FastAPI service on port 8001. The registered Traditional RAG graph loads onto the canvas at startup; it is also first in the template picker. Select the **Document Source** node to upload a TXT, Markdown, or selectable-text PDF, review the profile and chunk recommendation, choose a supported strategy, and index with Sentence Transformers. The divider above the playground can be dragged or focused and resized with the arrow keys to give more room to the graph or playground. Queries use persistent Chroma retrieval and your selected generation provider and display actual citations. The default vector store lives at `backend/data/chroma-recovered`; other gallery architectures remain metadata/planned and cannot execute.

The default embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. `sentence-transformers` and its PyTorch runtime are installed by `uv sync --group dev`; model weights are fetched from Hugging Face on first use unless already cached. It runs locally on CPU. To use Ollama embeddings instead, set `VRS_EMBEDDING_PROVIDER=ollama`, set `VRS_OLLAMA_EMBEDDING_MODEL`, and pull that model yourself. No provider fallback or mock result is used. The workflow Inspector's **Chunk Documents** node exposes only implemented character-based strategies/settings, and those graph values are the values used by indexing.

Groq is the default generation provider (`openai/gpt-oss-20b`). Select **Generate Answer** on the canvas (or add it from the palette) to set the provider/model, enter and save the Groq key, and test the provider. If no node is selected, the Inspector provides a **Configure Generate Answer** shortcut. The key is posted to the loopback backend over localhost and held in process memory; it is not saved in browser storage, written to disk, or returned by the API, and is cleared when the backend restarts or you click **Clear key**. Anyone with access to the local browser session or backend process can still use the configured key. Alternatively, set `GROQ_API_KEY` in the backend-only `.env` file. Ollama and Hugging Face remain selectable alternatives; Hugging Face uses backend-only `HF_TOKEN`. Keep embedding model IDs separate from chat model IDs. For Groq or Hugging Face questions, check the consent box directly above **Ask** before retrieved document excerpts can be sent to that external provider. **Ask** stays disabled until there is at least one indexed document, the selected provider has passed its test, and hosted-data consent is given. Hosted inference has provider usage limits and billing. See [Groq's API reference](https://console.groq.com/docs/api-reference) and [supported models](https://console.groq.com/docs/models).

Use the top navigation to open **Documents**, **Chunks**, and **Storage**. These views read the current SQLite/Chroma state from the backend; chunk results are paginated and can be filtered by document. The **Documents** view previews up to 20,000 characters of extracted source text. The workflow playground's query trace distinguishes retrieved chunks, assembled context preview, provider output, validated citations, and final answer. Query history is not persisted. Storage paths are intentionally returned as managed-local-path descriptions rather than arbitrary server paths.

Use the canvas toolbar to hide or restore the node palette and inspector. Canvas badges say **READY** when a component is implemented but idle; during indexing or answering they show the backend-reported node state. The active connection animates in execution order, and completed steps remain marked **DONE**. Drag empty canvas space to pan, scroll to move around the graph, and hold Ctrl while scrolling to zoom. The plus/minus and fit controls remain available on the canvas.

## Configuration and security

Copy `.env.example` to `.env`. Defaults bind to loopback and use local paths under `data/`. Non-loopback `VRS_HOST` values and `VRS_ENV=production` fail configuration validation unless `VRS_API_KEY` is set; clients then send it as `X-API-Key`. Browser access is limited to `VRS_CORS_ORIGINS` (default: the local Vite origins on port 5173). Read `docs/security.md` before exposing this server beyond loopback. `.env`, uploads, SQLite, Chroma files, model files, logs, caches, virtual environments and Node modules are ignored by Git.

Ollama URLs and embedding/generation model IDs are explicit configuration. Sentence Transformers downloads its selected model weights on first use unless `VRS_SENTENCE_TRANSFORMERS_LOCAL_FILES_ONLY=true`. No automatic provider failover, mock result, or in-memory vector-store fallback is allowed.

## Tests and checks

```powershell
cd backend
\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
\.venv\Scripts\ruff.exe check app tests --ignore B008
cd ..\frontend
npm test
npm run build
```

Backend tests cover health, registry/graph contracts, document parsing and profile evidence, chunking/provenance, Ollama adapter error behavior, SQLite and Chroma persistence/reopen, storage and chunk APIs, reindex idempotency, external-provider consent, and RAG API/runtime behavior. Frontend tests cover palette search/filtering, node/config editing, serialization, validation responses/errors, planned-template gating, explorer navigation, chunking metadata, and the Traditional RAG query trace. Mocked provider tests verify contracts only; they are not live-model evidence. See [Story 6 verification](docs/epics/story-6-storage-trace.md) for exact run results and limitations.

## Current limitations

- Workflow graph editing remains in-browser only; workflow persistence, imports/exports, and arbitrary graph execution are not implemented. Traditional RAG is a fixed validated graph executed by its deterministic runner.
- Document parsing supports UTF-8 text, Markdown, and PDFs with selectable text; OCR, layout reconstruction, and reliable language detection are not included. Chunk sizes use characters.
- Indexing requires the Sentence Transformers package and model weights (downloaded on first use by default), or an explicitly configured Ollama embedding model. Generation requires a separately configured chat model and provider; click **Test provider** to measure current inference availability. No real generation provider was configured in the verification workspace, so the template remains marked planned until a real-provider integration run passes.
- `/api/health/ready` remains a foundation-only check; use the Storage view or `/api/v1/storage/summary` for a live SQLite/Chroma check.
- A fresh setup needs package-registry access; this workspace currently has a resolved backend lock and installed dependencies.
- The RagSpark repository is under evaluation; see [library usage](docs/library-usage.md). Its README/reference claims have not been treated as runtime verification.
