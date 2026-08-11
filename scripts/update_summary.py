"""Update summary.json IN PLACE from a results dir, overwriting only the configs present
there and preserving every other record (heldout calibration, timings, fastrho-only configs
like anopheles_synth / real_drosophila, between_pop, etc.).

Usage: update_summary.py <summary.json> <results_dir> [--dry]

Used to fold the fair re-scores (auto-upRTR ReLERNN + pyrho bpen=40, in results_final/) into
the canonical summary the paper reads, without losing the records that were not re-scored.
"""
import json, sys, glob, os

summary_path, results_dir = sys.argv[1], sys.argv[2]
dry = "--dry" in sys.argv[3:]

s = json.load(open(summary_path))
updated = []
for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    name = os.path.basename(f)[:-5]
    if name == "summary":
        continue
    new = json.load(open(f))
    old = s.get(name, {})
    # A targeted re-score may intentionally include only a subset of methods.
    # Preserve method records that were not part of this run instead of dropping
    # unrelated baselines when the config record is refreshed.
    for scale, old_methods in old.get("scales", {}).items():
        new_methods = new.setdefault("scales", {}).setdefault(scale, {})
        for method, metrics in old_methods.items():
            new_methods.setdefault(method, metrics)
    s[name] = new
    # report the headline 25kb Pearson change per method for sanity
    def p25(rec):
        try:
            return {m: round(v.get("pearson", float("nan")), 3)
                    for m, v in rec["scales"]["25kb"].items()}
        except Exception:
            return {}
    updated.append((name, p25(old), p25(new)))

print(f"{'config':<20} {'old 25kb P':<34} -> new 25kb P")
for name, o, n in updated:
    print(f"{name:<20} {str(o):<34} -> {n}")
if not dry:
    json.dump(s, open(summary_path, "w"), indent=2)
    print(f"\nwrote {summary_path} ({len(updated)} configs updated, others preserved)")
else:
    print("\n[dry run] not written")
