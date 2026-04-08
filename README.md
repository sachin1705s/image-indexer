# Image Indexer

A local-first image search tool for your own folders.

Point it at a folder of images, let it build a local index, then search your archive in plain English from a localhost web UI.

Examples:

```text
dragon
warm editorial portrait
minimal product screenshot
blue landscape with negative space
```

No cloud upload. No external APIs. Your images stay on your machine.

## What It Does

- Indexes local image folders recursively.
- Supports `.jpg`, `.jpeg`, and `.png`.
- Creates local thumbnails for fast browsing.
- Generates local CLIP embeddings for text search and image similarity.
- Adds simple local auto-labels like content type, style tags, and object tags.
- Lets you search from a browser at `http://127.0.0.1:8000`.
- Lets you upload an image to find visually similar images.
- Skips unchanged files when you index the same folder again.

## How To Use

Install/run it directly from PyPI:

```bash
uvx --from image-archive-search image-archive-search run
```

If you do not have `uvx` yet:

```bash
pip install uv
uvx --from image-archive-search image-archive-search run
```

The `run` command will:

1. Create local app data if needed.
2. Open a terminal folder picker.
3. Let you select one or more folders.
4. Index supported images.
5. Offer to start the local web app.

When the server starts, open:

```text
http://127.0.0.1:8000
```

That is the main workflow. You do not need to clone this repo to use the app.

## Folder Picker

Inside the terminal picker:

```text
↑ / ↓     move
Space     select or unselect a folder
Enter     open folder
→         open folder
←         go up
d         done, start indexing
q         cancel
```

## Other User Commands

```bash
# Guided setup, folder selection, indexing, and optional server start
uvx --from image-archive-search image-archive-search run

# Index one folder directly
uvx --from image-archive-search image-archive-search index ~/Pictures

# Start the local web app for already-indexed images
uvx --from image-archive-search image-archive-search serve

# Show archive status
uvx --from image-archive-search image-archive-search status

# Recheck configured folders and index only changed/missing files
uvx --from image-archive-search image-archive-search reindex

# Delete local index data, thumbnails, and vectors
# This does not delete your original images.
uvx --from image-archive-search image-archive-search reset
```

## What Gets Stored

For each indexed image, the app stores local metadata:

- file path
- file hash
- width and height
- created and modified timestamps when available
- thumbnail path
- local search embedding
- local labels/tags

The original image files are not modified.

By default, app data is stored in your user app-data folder:

```text
macOS:   ~/Library/Application Support/image-archive-search/
Linux:   ~/.local/share/image-archive-search/
Windows: %LOCALAPPDATA%/image-archive-search/
```

## Why Local-First?

This tool is meant for personal and creative archives where privacy matters.

- Your images are indexed locally.
- Search runs locally.
- The web UI is served from localhost.
- No account is required.
- No cloud image upload is required.

The first run may download local model weights, then reuse them from your machine afterward.

## Current Limitations

- Images only: `.jpg`, `.jpeg`, `.png`.
- No video indexing.
- No face recognition or identity detection.
- CLIP labels are useful but not as detailed as a large vision-language model.
- Very large archives may need future large-library optimizations.
- Deleted files are not fully cleaned up automatically yet.

## Development

Clone the repo and install locally:

```bash
git clone https://github.com/sachin1705s/image-indexer.git
cd image-indexer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q
```

Run the app from source:

```bash
image-archive-search run
```

## Project Structure

```text
backend/image_archive/   Python package, CLI, API, indexing, search
backend/image_archive/frontend/
                         Packaged localhost web UI
frontend/                Source copy of the minimal UI
tests/                   Test suite
config.example.yaml      Example config
pyproject.toml           Python package metadata
```

## Publishing

This package is published on PyPI as:

```text
image-archive-search
```

Build and upload a new release:

```bash
python -m pip install --upgrade build twine
rm -rf dist build *.egg-info backend/*.egg-info
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
python -m twine upload dist/*.whl dist/*.tar.gz
```

Before uploading a new release, bump the version in `pyproject.toml`.
