"""Runtime environment preparation for local ML dependencies."""

from __future__ import annotations

import os
import platform


def prepare_runtime_environment() -> None:
    """Apply conservative runtime environment tweaks before ML libs import.

    On some macOS Python setups, FAISS and Torch can each load an OpenMP runtime,
    which aborts the process before the app starts. This workaround is not ideal,
    but it keeps the local MVP usable on those machines.
    """

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if platform.system() == "Darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
