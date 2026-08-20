"""Load the locked authoritative manuscript staged by ``reproduce/``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_ROOT = Path(
    os.environ.get("FASTRHO_MANUSCRIPT_ROOT", ROOT / "tmp" / "reproduce" / "manuscript")
).resolve()
MAIN_PATH = MANUSCRIPT_ROOT / "main.tex"
SI_PATH = MANUSCRIPT_ROOT / "si.tex"

if not MAIN_PATH.is_file() or not SI_PATH.is_file():
    pytest.skip(
        "locked manuscript is not staged; run reproduce/fetch_manuscript.py",
        allow_module_level=True,
    )

MAIN = MAIN_PATH.read_text(encoding="utf-8")
SI = SI_PATH.read_text(encoding="utf-8")
