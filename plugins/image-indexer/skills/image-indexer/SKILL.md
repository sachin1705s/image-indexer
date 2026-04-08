---
name: image-indexer
description: Help users install, run, troubleshoot, reset, package, and publish Image Indexer, the local-first image archive search CLI.
---

# Image Indexer

Use this skill when the user asks about the Image Indexer CLI, including installing it from PyPI, running the local web UI, indexing folders, resetting the index, debugging CLI issues, or publishing a new release.

## Product Summary

Image Indexer is a local-first image search tool. It indexes `.jpg`, `.jpeg`, and `.png` files from local folders, stores metadata in SQLite, stores vectors in FAISS, creates thumbnails, and serves a localhost web UI.

The main user command is:

```bash
uvx --from image-archive-search image-archive-search run
```

If `uvx` is missing:

```bash
pip install uv
uvx --from image-archive-search image-archive-search run
```

## Common Commands

```bash
uvx --from image-archive-search image-archive-search run
uvx --from image-archive-search image-archive-search serve
uvx --from image-archive-search image-archive-search status
uvx --from image-archive-search image-archive-search reindex
uvx --from image-archive-search image-archive-search reset
```

For local repo development:

```bash
cd /Users/sachin/Documents/image-indexer
source ~/.venv/bin/activate
image-archive-search run
```

## Folder Picker

Tell users:

```text
↑ / ↓     move
Space     select or unselect a folder
Enter     open folder
→         open folder
←         go up
d         done, start indexing
q         cancel
```

## Local Data Paths

Default app data lives outside the repo:

```text
macOS:   ~/Library/Application Support/image-archive-search/
Linux:   ~/.local/share/image-archive-search/
Windows: %LOCALAPPDATA%/image-archive-search/
```

Repo-local dev configs may use:

```text
/Users/sachin/Documents/image-indexer/config.yaml
/Users/sachin/Documents/image-indexer/.image-archive/
```

If the UI shows indexed images but the repo-local archive looks empty, check whether a server was started from another directory or is using the default app-data path.

## Troubleshooting

- If `uvx: command not found`, suggest `pip install uv`.
- If the web UI is not visible, suggest `image-archive-search serve` and open `http://127.0.0.1:8000`.
- If indexing appears stale, suggest `image-archive-search reindex`.
- If the user wants to delete index data, use `image-archive-search reset`; remind them it does not delete original images.
- If macOS shows an OpenMP/libomp crash, suggest restarting through the CLI first. The app applies a runtime workaround. If needed, use `KMP_DUPLICATE_LIB_OK=TRUE`.
- If images show duplicates, explain that the app indexes physical file paths separately.
- If video frames show up, check excluded folders and path patterns before reindexing.

## Publishing

For a new PyPI release:

```bash
cd /Users/sachin/Documents/image-indexer
source ~/.venv/bin/activate
rm -rf dist build *.egg-info backend/*.egg-info
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
python -m twine upload dist/*.whl dist/*.tar.gz
```

Always bump `version` in `pyproject.toml` before publishing. PyPI does not allow reuploading the same version.

## GitHub

Public repo:

```text
https://github.com/sachin1705s/image-indexer
```

Published package:

```text
https://pypi.org/project/image-archive-search/
```
