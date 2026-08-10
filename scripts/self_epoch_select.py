"""Real-data epoch selection for the selfing model (and any real-targeted model).

Root-cause fix for the 15k-self regression. The `self` checkpoint was picked on the SIMULATED
val_pearson (epoch 36, 0.963 -- the highest in the whole zoo), but that criterion overfits the
selfing *simulator*: real A. thaliana recovery fell 0.27 (self2) -> 0.11 (self15k) as sim val_loss
improved 0.109 -> -0.052. Selecting on simulated validation is structurally blind to the sim->real
gap. This scores EVERY saved epoch checkpoint on real A. thaliana chromosomes and selects on a
HELD-OUT chromosome, so the pick is made on the real target -- mirroring the rich model's CEU
real-proxy early-stopping (which landed on epoch 2/40 for exactly this reason).

Scoring is identical to scripts/realdata_infer.py (same 100 kb windows, same stdpopsim truth,
same Pearson) so the numbers are directly comparable to the ED selfer figure's per-chrom r.

Usage (on sesame, fastrho venv):
  python scripts/self_epoch_select.py <ckpt_glob> <feat_stats.npz> \
      [--select 1] [--report 2,3,4,5] [--out best_ckpt.txt]
  # ckpt_glob e.g. '/home/kkor/fastrho_data/campaign_self/train_allep/fastrho/version_*/checkpoints/*.ckpt'
"""
import sys, os, glob, re, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from scipy.stats import pearsonr
from fastrho.translate import load_model, predict_map_from_genotype_matrix, predict_from_tokens
import realdata_infer as RI   # reuse truth_windows + the exact windowing constants

HAP = "/home/kkor/realdata/hap"
W = RI.W          # 100 kb, same as realdata_infer
DEV = RI.DEV
CACHE_DIR = None  # set from --cache-dir; when set, score_chrom reuses cached raw tokens


def _epoch_of(path):
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def score_chrom(model, cfg, stats, chrom):
    """Real A. thaliana recovery for one chromosome; identical pipeline to realdata_infer.main().
    With CACHE_DIR set, reuse pre-computed raw tokens (model-independent) and only run the GPU
    forward -- turns a 50-epoch sweep from ~50x featurization into 1x."""
    cache = os.path.join(CACHE_DIR, f"athal_c{chrom}_tokens.npz") if CACHE_DIR else None
    if cache and os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        pred = predict_from_tokens(model, cfg, stats, c["tokens"], c["positions"].astype(np.float64),
                                   int(c["n_hap"]), float(c["mu"]), Ne=None, device=DEV)
        chrom_id = str(c["chrom"]); map_sp = str(c["map_sp"]); map_id = str(c["map_id"])
    else:
        z = np.load(f"{HAP}/athal_c{chrom}.npz", allow_pickle=True)
        gm = z["gm"]; pos = z["pos"].astype(np.float64)
        mu = float(z["mu"]); chrom_id = str(z["chrom"])
        map_sp = str(z["map_sp"]); map_id = str(z["map_id"])
        pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                                mutation_rate=mu, Ne=None, device=DEV)
    left = pred["pos_left"]; right = pred["pos_right"]
    bp = np.r_[left[0], right]
    lo = int(left[0]); hi = int(right[-1])
    edges = np.append(np.arange(lo, hi, W), hi)
    pred_r = RI.mean_rate_between(bp, pred["r_per_bp"], edges)
    true_r = RI.truth_windows(map_sp, map_id, chrom_id, edges)
    k = min(len(pred_r), len(true_r))
    p, t = pred_r[:k], true_r[:k]
    ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
    return float(pearsonr(p[ok], t[ok])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_glob")
    ap.add_argument("feat_stats")
    ap.add_argument("--select", default="1", help="chrom(s) to SELECT on (held-out), comma-sep")
    ap.add_argument("--report", default="2,3,4,5", help="chrom(s) to REPORT on, comma-sep")
    ap.add_argument("--topk", default=5, help="how many top select-scorers to score on report chroms")
    ap.add_argument("--cache-dir", default=None, help="dir of self_featurize_cache.py token caches")
    ap.add_argument("--out", default=None, help="write the winning checkpoint path here")
    a = ap.parse_args()
    global CACHE_DIR
    CACHE_DIR = a.cache_dir
    sel = [c.strip() for c in a.select.split(",") if c.strip()]
    rep = [c.strip() for c in a.report.split(",") if c.strip()]
    all_chroms = sorted(set(sel + rep), key=int)

    topk = int(a.topk)
    ckpts = sorted(glob.glob(a.ckpt_glob), key=_epoch_of)
    if not ckpts:
        sys.exit(f"no checkpoints match {a.ckpt_glob}")
    print(f"{len(ckpts)} checkpoint(s) | SELECT on chr{','.join(sel)} (held-out) | "
          f"REPORT on chr{','.join(rep)} | two-phase (top-{topk})", flush=True)

    # --- phase 1: score every epoch on the held-out SELECT chrom(s) only (cheap) ---
    print("phase 1 -- select-chrom scan:", flush=True)
    p1 = []
    for ck in ckpts:
        model, cfg, stats = load_model(ck, a.feat_stats, device=DEV)
        rs = {c: score_chrom(model, cfg, stats, c) for c in sel}
        sm = float(np.mean(list(rs.values())))
        p1.append((_epoch_of(ck), ck, rs, sm))
        print(f"  epoch {_epoch_of(ck):>3}  sel={sm:+.3f}  " +
              " ".join(f"chr{c}={rs[c]:+.2f}" for c in sel), flush=True)

    # --- phase 2: score only the top-k select-scorers on the REPORT chroms ---
    cand = sorted(p1, key=lambda x: x[3], reverse=True)[:topk]
    print(f"phase 2 -- report-chrom scoring of top {len(cand)} candidate(s):", flush=True)
    scored = []
    for ep, ck, rs, sm in cand:
        model, cfg, stats = load_model(ck, a.feat_stats, device=DEV)
        rr = {c: score_chrom(model, cfg, stats, c) for c in rep}
        full = {**rs, **rr}
        rm = float(np.mean([full[c] for c in rep]))
        scored.append((ep, ck, full, sm, rm))
        print(f"  epoch {ep:>3}  sel={sm:+.3f}  report={rm:+.3f}  " +
              " ".join(f"chr{c}={full[c]:+.2f}" for c in all_chroms), flush=True)

    # winner = best held-out select score; report is its honest out-of-selection performance
    ep, ck, full, sm, rm = max(scored, key=lambda x: x[3])
    print("=" * 60)
    print(f"WINNER  epoch={ep}  select_mean={sm:+.3f}  report_mean={rm:+.3f}")
    print(f"  all5_mean={np.mean([full[c] for c in all_chroms]):+.3f}  "
          f"per-chrom={json.dumps({c: round(full[c], 3) for c in all_chroms})}")
    print(f"  ckpt: {ck}")
    print(f"  incumbent self2 all5_mean=+0.270 ; overfit self15k all5_mean=+0.110")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(ck + "\n")
        print(f"  wrote winner -> {a.out}")


if __name__ == "__main__":
    main()
