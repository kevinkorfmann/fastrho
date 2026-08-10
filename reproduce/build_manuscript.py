#!/usr/bin/env python3
"""Compile the staged canonical Phase 2 manuscript without touching its repository."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / "tmp" / "reproduce" / "manuscript"
OUTPUT = ROOT / "output" / "pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    args = parser.parse_args()
    manuscript = args.manuscript.expanduser().resolve()
    environment = dict(os.environ)
    environment.setdefault("SOURCE_DATE_EPOCH", "1420070400")
    for source in ("si_phase2.tex", "main_phase2.tex"):
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", source],
            cwd=manuscript,
            env=environment,
            check=True,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manuscript / "main_phase2.pdf", OUTPUT / "fastrho_manuscript_phase2.pdf")
    shutil.copy2(manuscript / "si_phase2.pdf", OUTPUT / "fastrho_manuscript_phase2_si.pdf")
    print(OUTPUT / "fastrho_manuscript_phase2.pdf")
    print(OUTPUT / "fastrho_manuscript_phase2_si.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
