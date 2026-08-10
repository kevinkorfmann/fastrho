"""Evaluate one blindly frozen member on preregistered simulation strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastrho.evaluate import evaluate_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    results = {}
    for path in sorted(item for item in args.audit_root.iterdir() if item.is_dir()):
        results[path.name] = evaluate_dir(
            args.checkpoint,
            args.stats,
            str(path),
            device=args.device,
            scales=(100_000,),
            hotspot_scale=100_000,
            ne_mode="both",
        )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stats": str(Path(args.stats).resolve()),
        "selection_data": "held-out simulated audit only",
        "arabis_cross_map_used": False,
        "strata": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            mode: values[mode].get("100kb", {}).get("pearson")
            for mode in ("true", "estimated")
        }
        for name, values in results.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
