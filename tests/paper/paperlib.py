"""paperlib — the re-derivation engine for fastrho's Phase 2 result tests.

Philosophy
----------
The active prose-wide audit is ``reproduce/audit_phase2.py``. This module keeps
the lower-level result re-derivations used by focused tests: wherever raw paired
(prediction, truth) arrays are committed, snapshot metrics are recomputed with
the same estimator used by the analysis pipeline.

This module is intentionally dependency-light: numpy + scipy only (sklearn is
optional, used only for AUPRC). It imports nothing from torch, so the whole
paper-number suite runs on any machine, GPU or not.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
# tests/paper/paperlib.py  ->  parents[0]=paper  [1]=tests  [2]=<repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper"
BIBLIOGRAPHY = REPO_ROOT / "refs.bib"
SNAPSHOT = PAPER / "results_snapshot"
FIGDATA = PAPER / "figdata"
TABLES = PAPER / "tables"
RESULTS = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Canonical-data loaders (cached)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def load_json(path_str: str) -> dict:
    with open(path_str) as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def summary() -> dict:
    """The master results record: paper/results_snapshot/summary.json."""
    return load_json(str(SNAPSHOT / "summary.json"))


@functools.lru_cache(maxsize=None)
def snapshot(name: str) -> dict:
    """Any other results_snapshot/<name>.json record."""
    if not name.endswith(".json"):
        name += ".json"
    return load_json(str(SNAPSHOT / name))


@functools.lru_cache(maxsize=None)
def figjson(name: str) -> dict:
    if not name.endswith(".json"):
        name += ".json"
    return load_json(str(FIGDATA / name))


def load_npz(path: Path) -> dict:
    """Load an .npz into a plain dict (arrays kept as arrays)."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Navigate the snapshot by dotted path
# ---------------------------------------------------------------------------
def get(path: str, root: dict | None = None):
    """Fetch a value from the summary by dotted path.

    e.g. get("const_n20.scales.100kb.fastrho.pearson")
    Raises KeyError with a helpful message if any segment is missing.
    """
    node = summary() if root is None else root
    walked = []
    for seg in path.split("."):
        walked.append(seg)
        if not isinstance(node, dict) or seg not in node:
            raise KeyError(f"snapshot path not found at {'.'.join(walked)!r} "
                           f"(full path {path!r})")
        node = node[seg]
    return node


# Convenience accessor matching the snapshot's natural shape.
def metric(config: str, scale: str, method: str, name: str):
    """summary[config][scales][scale][method][name]"""
    return get(f"{config}.scales.{scale}.{method}.{name}")


CONFIGS = (
    "anopheles_synth", "between_pop_d00", "between_pop_d50",
    "bottleneck_n20", "bottleneck_n20_wd", "const_n100", "const_n20",
    "const_n40", "expansion_n20", "heldout", "real_decode", "real_dog",
    "real_drosophila", "real_hapmap",
)
# Configs that carry the standard per-scale per-method metric vectors.
BENCH_CONFIGS = (
    "anopheles_synth", "bottleneck_n20", "bottleneck_n20_wd", "const_n100",
    "const_n20", "const_n40", "expansion_n20", "real_decode", "real_dog",
    "real_drosophila", "real_hapmap",
)
SCALES = ("25kb", "100kb", "500kb")
METHODS = ("fastrho", "pyrho", "gruseq2seq", "relernn")
METRIC_NAMES = ("pearson", "spearman", "log_pearson", "l2", "log_l2",
                "bias_ratio", "hotspot_auprc")


# ---------------------------------------------------------------------------
# Metric estimator — EXACT mirror of fastrho.evaluate.score_rates
# ---------------------------------------------------------------------------
def score_rates(pred, true) -> dict:
    """Re-implementation of fastrho.evaluate.score_rates, kept byte-for-byte in
    sync so we re-derive the snapshot with the identical estimator.

    Masks to finite & strictly-positive pairs, then computes the metric vector.
    """
    from scipy.stats import pearsonr, spearmanr
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    p, t = pred[m], true[m]
    if len(p) < 3:
        return {"n": int(m.sum())}
    return {
        "pearson": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t)[0]),
        "log_pearson": float(pearsonr(np.log(p), np.log(t))[0]),
        "l2": float(np.sqrt(np.mean((p - t) ** 2))),
        "log_l2": float(np.sqrt(np.mean((np.log(p) - np.log(t)) ** 2))),
        "bias_ratio": float(np.median(p / t)),
        "n": int(m.sum()),
    }


def pearson(pred, true) -> float:
    from scipy.stats import pearsonr
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    return float(pearsonr(pred[m], true[m])[0])


# ---------------------------------------------------------------------------
# Rounding-aware comparison
# ---------------------------------------------------------------------------
def _decimals_of(written: str) -> int:
    """Number of decimal places in a written numeric literal (after stripping
    sign, thousands separators, and a trailing %)."""
    s = written.strip().lstrip("+-").replace(",", "").rstrip("%")
    if "e" in s.lower():
        # scientific notation — treat mantissa decimals
        s = s.lower().split("e")[0]
    return len(s.split(".")[1]) if "." in s else 0


def _token_and_scale(written: str):
    """Return (numeric token as written, factor to bring `actual` onto the same
    scale). For a percent literal the token stays in percent units and the
    factor is 100, so half-ULP comparisons happen on the printed scale."""
    s = (written.strip()
         .replace("≈", "").replace("~", "").replace("$", "")
         .replace("\\,", "").replace(",", "").strip())
    pct = s.endswith("%")
    s = s.rstrip("%")
    return float(s), (100.0 if pct else 1.0)


def parse_number(written: str) -> float:
    """Parse a written number as a fraction (percent -> /100)."""
    tok, scale = _token_and_scale(written)
    return tok / scale


def matches_rounded(actual: float, written: str) -> bool:
    """True iff `actual` rounds to the literal `written` at the precision
    `written` is quoted to. Implemented as a half-ULP interval test (on the
    printed scale, percent-aware) rather than Python's round(), so hand-rounded
    half-way values (a prose "0.58" for an actual 0.575) pass and banker's
    rounding never bites:  |actual·scale - token| <= 0.5·10^-decimals + eps."""
    tok, scale = _token_and_scale(written)
    nd = _decimals_of(written)
    return abs(float(actual) * scale - tok) <= 0.5 * 10 ** (-nd) + 1e-9


def approx(actual: float, written: str, rel: float = 0.0, atol: float = 0.0) -> bool:
    """Approximate match for quoted-with-~ quantities (speedups, fold-changes).
    Defaults to a 2% relative tolerance OR half-ULP of the written precision,
    whichever is looser, unless rel/atol are supplied. Percent-aware."""
    tok, scale = _token_and_scale(written)
    a = float(actual) * scale
    if rel == 0.0 and atol == 0.0:
        rel = 0.02
        atol = 0.5 * 10 ** (-_decimals_of(written))
    return abs(a - tok) <= max(atol, rel * abs(tok))


# ---------------------------------------------------------------------------
# Derived quantities the paper reports
# ---------------------------------------------------------------------------
def all_pearsons(method: str, scale: str, configs=BENCH_CONFIGS):
    """Pearson values for `method` at `scale` across the given configs that
    have that cell (skips missing cells, e.g. relernn only exists in some)."""
    out = []
    for c in configs:
        try:
            v = metric(c, scale, method, "pearson")
        except KeyError:
            continue
        out.append((c, v))
    return out


def fastrho_wallclock():
    """{config: fastrho wall_clock_s} across all benchmark configs that record it."""
    out = {}
    for c in BENCH_CONFIGS:
        rec = summary().get(c, {})
        wc = rec.get("wall_clock_s", {})
        if "fastrho" in wc:
            out[c] = wc["fastrho"]
    return out


def win_count(scale: str, a: str = "fastrho", b: str = "pyrho",
              metric_name: str = "pearson", configs=BENCH_CONFIGS):
    """Number of configs (with both cells) where method `a` beats `b`, and the
    total number of comparable configs."""
    wins = total = 0
    for c in configs:
        try:
            va = metric(c, scale, a, metric_name)
            vb = metric(c, scale, b, metric_name)
        except KeyError:
            continue
        total += 1
        wins += int(va > vb)
    return wins, total


def speedup(method: str) -> float:
    """Wall-clock cost normalized to fastrho, from the timings record."""
    t = summary()["timings"]
    return t[method] / t["fastrho"]


# ---------------------------------------------------------------------------
# LaTeX table parsing
# ---------------------------------------------------------------------------
def read_tex(name: str) -> str:
    p = name if Path(name).is_absolute() else str(TABLES / name)
    with open(p) as fh:
        return fh.read()


def tex_rows(tex: str):
    """Yield (cells) for each data row of a tabular: split on '\\\\', drop the
    preamble/rules, split each row on '&', strip latex cruft per cell."""
    # remove comments
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    body = tex
    for marker in (r"\toprule", r"\midrule", r"\bottomrule"):
        body = body.replace(marker, "")
    for raw in body.split(r"\\"):
        raw = raw.strip()
        if not raw or "tabular" in raw:
            continue
        if "&" not in raw:
            continue
        cells = [clean_cell(c) for c in raw.split("&")]
        yield cells


def clean_cell(cell: str) -> str:
    c = cell.strip()
    c = c.replace(r"\textbf", "").replace(r"\emph", "")
    c = c.replace(r"\quad", "").replace(r"\multicolumn", "")
    c = re.sub(r"\\[a-zA-Z]+", "", c)        # strip remaining latex macros
    c = c.replace("{", "").replace("}", "")
    c = c.replace(r"\,", "").replace("$", "").replace("\\", "")
    c = c.replace("−", "-")             # unicode minus
    return c.strip()


def between_pop(diff: str) -> dict:
    """The between-population record. diff in {'d00','d50'}."""
    return snapshot(f"between_pop_{diff}")


FIGURES = PAPER / "figures"


@functools.lru_cache(maxsize=None)
def stdpopsim_pearsons(variant: str):
    """Per-species Pearson values from paper/figures/_stdpopsim_<variant>.json
    (the phasing/polarization ablation panels). Returns a list of floats."""
    data = load_json(str(FIGURES / f"_stdpopsim_{variant}.json"))
    out = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "pearson" and isinstance(v, (int, float)):
                    out.append(float(v))
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return out


@functools.lru_cache(maxsize=None)
def agam_concordance_meta() -> dict:
    """Per-cohort meta embedded in figdata/agam_pyrho_concordance_arrays.npz."""
    z = load_npz(FIGDATA / "agam_pyrho_concordance_arrays.npz")
    return json.loads(str(z["meta"]))


@functools.lru_cache(maxsize=None)
def showdown_meta() -> dict:
    """meta dict embedded in figdata/relernn_showdown.npz (dots, curve, grids_kb,
    hotspot geometry)."""
    z = load_npz(FIGDATA / "relernn_showdown.npz")
    return json.loads(str(z["meta"]))


# ---- tree-of-life (pyrho head-to-head) table ------------------------------
# The committed pyrho_headtohead.tex IS the canonical source for these numbers
# (the upstream per-species run lives on the compute host, not in the repo), so
# we treat the parsed table as ground truth and check the prose against it.
def _headtohead_sections():
    """Split pyrho_headtohead.tex into ('simulated', rows) and ('real', rows).
    Both sections contain a 'Human' row, so they must be parsed separately."""
    tex = read_tex("pyrho_headtohead.tex")
    # The "Real genotypes" \multicolumn header marks the boundary.
    head, _, tail = tex.partition("Real genotypes")
    sim, real = [], []
    for chunk, bucket in ((head, sim), (tail, real)):
        for cells in tex_rows(chunk):
            if len(cells) < 4:
                continue
            label = clean_cell(cells[0])
            f, p = as_number(cells[-2]), as_number(cells[-1])
            if label and f is not None and (p is not None or cells[-1].strip() in ("---", "--", "—")):
                bucket.append((label, f, p))
    return sim, real


def tree_of_life_rows():
    """[(label, fastrho, pyrho)] for the 7 simulated 'tree of life' species."""
    return _headtohead_sections()[0]


def real_genotype_rows():
    """[(label, fastrho, pyrho)] for the real-genotype rows (pyrho may be None
    for the unphased wolf/dog rows)."""
    return _headtohead_sections()[1]


_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def as_number(cell: str):
    """Return float(cell) if the cleaned cell is a plain number, else None
    (handles '--', '—', 'config' headers, etc.)."""
    c = cell.replace(",", "").replace("kb", "").strip()
    c = c.replace("—", "").replace("--", "").strip()
    if _NUM_RE.match(c):
        return float(c)
    return None
