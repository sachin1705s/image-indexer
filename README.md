# Image Archive Search

Local-first creative image archive search for personal libraries.

This project indexes one or more folders of images on your machine, generates local thumbnails, CLIP zero-shot enrichment labels, and CLIP-style embeddings, stores metadata in SQLite, stores vectors in FAISS, and serves a localhost web UI for natural-language and image-to-image search.

## MVP v1 Features

- Local-first only. No external APIs or cloud services.
- Index local image folders recursively.
- Store file path, hash, dimensions, timestamps, folder, thumbnail, embedding metadata, and structured enrichment fields.
- Incremental indexing that skips unchanged files and resumes cleanly after interruptions.
- Text search with embedding retrieval plus structured tag/style/object boosting.
- Image-to-image similarity search from an indexed asset or uploaded query image.
- Folder and date filtering.
- Content-type filtering.
- Similar-images view for any asset.
- Guided CLI workflow with `run`, plus power-user commands `init`, `index`, `serve`, `status`, and `reindex`.
- Installable CLI shape with the `image-archive-search` command, packaged frontend assets, per-user app data, and a `reset` command.

## Supported File Types

- `.jpg`
- `.jpeg`
- `.png`

## Project Structure

```text
backend/   Python package, CLI, API, indexing pipeline, search services
frontend/  Minimal local web UI served by FastAPI
models/    Notes and placeholders for local model assets
scripts/   Helper scripts
tests/     Basic test suite
```

## How It Works

1. `init` creates a local app data directory, SQLite DB, FAISS index, and config file.
2. `run` or `index` scans image files, skips unchanged assets, creates thumbnails, embeddings, and CLIP zero-shot enrichment fields, then persists everything locally.
3. `serve` launches the FastAPI server and serves the UI from `http://127.0.0.1:8000`.
4. The UI lets you search in plain English, filter by indexed folder, content type, or date, upload a query image, and inspect similar results.

## Install And Run

For users, the intended packaged command is:

```bash
uvx --from image-archive-search image-archive-search run
```

For a permanent install:

```bash
uv tool install image-archive-search
image-archive-search run
```

During local development from this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
image-archive-search run
```

The guided command initializes the app, opens the terminal folder picker, indexes selected folders, and can start the localhost UI.

## Local Development Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

Notes:

- The first time you run indexing, the embedding model weights will be downloaded locally and reused from cache afterward.
- Structured enrichment defaults to CLIP zero-shot labels and does not require Ollama.
- Ollama remains an optional backend for richer VLM enrichment if you set `enrichment_backend: ollama`.
- On some Apple Silicon or Linux setups, `faiss-cpu` may be easiest to install through `conda` if a wheel is not available for your environment.
- On some macOS setups, OpenMP libraries from FAISS and Torch can conflict. The CLI applies a compatibility workaround automatically, but if you still see an OpenMP startup error, run commands with `KMP_DUPLICATE_LIB_OK=TRUE`.

### 3. Initialize the archive

```bash
image-archive-search init
```

By default this creates per-user files outside the repo:

- macOS config/data: `~/Library/Application Support/image-archive-search/`
- Linux data: `~/.local/share/image-archive-search/`
- Linux config: `~/.config/image-archive-search/config.yaml`
- Windows config/data: under `%APPDATA%` and `%LOCALAPPDATA%`

You can still force a repo-local config for development:

```bash
image-archive-search init --config-path config.yaml
```

### 4. Guided flow

```bash
image-archive-search run
```

This guided command:

- initializes the local archive if needed
- opens a terminal folder navigator
- lets you multi-select folders to index
- runs the full indexing pipeline
- optionally starts the local server

### 5. Index a folder directly

```bash
image-archive-search index /path/to/library
```

You can run `index` again on the same folder. Unchanged files are skipped automatically.

### 6. Serve the local app

```bash
image-archive-search serve
```

Then open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

## CLI Commands

```bash
image-archive-search init
image-archive-search run
image-archive-search index /path/to/library
image-archive-search reindex
image-archive-search status
image-archive-search serve --host 127.0.0.1 --port 8000
image-archive-search reset
```

The legacy `image-archive` command remains available. New users should prefer `image-archive-search`.

## Example Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

image-archive-search run
```

## Publishing To PyPI

1. Pick a final package name on PyPI. The current package name is `image-archive-search`.
2. Build the wheel and source distribution:

```bash
python3 -m pip install --upgrade build twine
python3 -m build
```

3. Check the package:

```bash
python3 -m twine check dist/*
```

4. Publish to TestPyPI first:

```bash
python3 -m twine upload --repository testpypi dist/*
```

5. Test install from TestPyPI in a clean environment.

6. Publish to PyPI:

```bash
python3 -m twine upload dist/*
```

After publishing, users can run:

```bash
uvx --from image-archive-search image-archive-search run
```

## Configuration

The default config is created by `init` in the per-user app config directory. A sample is also provided as [`config.example.yaml`](/Users/sachin/Documents/image-indexer/config.example.yaml).

Key fields:

- `indexed_paths`
- `thumbnail_dir`
- `sqlite_path`
- `faiss_index_path`
- `embedding_model_name`
- `enrichment_backend`
- `enrichment_model`
- `enrichment_mode`
- `enrichment_version`
- `ollama_host`
- `device`
- `batch_size`
- `num_workers`

## Search Behavior

- Text search embeds the query with the local embedding model, retrieves nearest vectors from FAISS, and boosts results whose content type, tags, styles, objects, and short summaries match the query.
- Similar search uses the indexed asset embedding or a locally uploaded image.
- Exact self-matches are excluded from similar results by default.

## Limitations

- MVP v1 supports images only. Video is intentionally out of scope.
- CLIP zero-shot enrichment is fast but less nuanced than a larger VLM for OCR-heavy document analysis and detailed object reasoning.
- Index updates currently focus on new, changed, stale, or partially processed records. Automatic deletion handling for files removed from disk is minimal in v1.
- The first indexing run can be slow because local models are loaded and warmed up.
- The UI is intentionally minimal and optimized for usability over design polish.

## Future Roadmap

- Better reranking and search-time faceting
- Duplicate clustering
- Richer asset facets and saved collections
- Video, OCR, and extra metadata extractors
- Faster background indexing workers
- Model selection from the UI

## Repo Tree

```text
.
|-- backend/
|   `-- image_archive/
|-- frontend/
|-- models/
|-- scripts/
|-- tests/
|-- config.example.yaml
|-- pyproject.toml
`-- README.md
```

## Commands To Run Locally

```bash
image-archive-search init
image-archive-search run
image-archive-search serve
```

Or with the packaged command:

```bash
image-archive-search run
image-archive-search serve
image-archive-search reset
```

## Known Limitations

- Removed files are not fully garbage-collected from search results in every case yet.
- Embeddings and CLIP zero-shot enrichment depend on local model downloads.
- Very large archives may benefit from future background jobs and sharded indexing.
