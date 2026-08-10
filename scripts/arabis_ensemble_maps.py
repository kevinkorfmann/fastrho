"""Average independently trained, blindly frozen Arabis prediction maps."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--expected-files", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(args.manifest)
    reference_dir = args.input_root / "seed0"
    files = sorted(reference_dir.glob("*.npz"))
    if len(files) != args.expected_files:
        raise RuntimeError(
            f"expected {args.expected_files} maps, found {len(files)} in {reference_dir}"
        )
    for reference_path in files:
        paths = [args.input_root / f"seed{i}" / reference_path.name for i in range(args.members)]
        loaded = [np.load(path) for path in paths]
        try:
            for key in ("pos_left", "pos_right"):
                for candidate in loaded[1:]:
                    np.testing.assert_array_equal(loaded[0][key], candidate[key])
            output = {key: loaded[0][key] for key in loaded[0].files}
            log_rates = np.stack([z["log_rho"] for z in loaded])
            log_mean = np.mean(log_rates, axis=0)
            sigma = np.stack([z["sigma_log_rho"] for z in loaded])
            log_variance = np.maximum(
                0.0, np.mean(sigma**2 + log_rates**2, axis=0) - log_mean**2
            )
            sigma_total = np.sqrt(log_variance)
            ne_estimated = float(np.exp(np.mean(np.log([float(z["Ne_estimated"]) for z in loaded]))))
            ne_used = float(np.exp(np.mean(np.log([float(z["Ne_used"]) for z in loaded]))))
            output.update({
                "log_rho": log_mean,
                "sigma_log_rho": sigma_total,
                "rho_per_bp": np.exp(log_mean),
                "rho_ci_lo": np.exp(log_mean - 1.96 * sigma_total),
                "rho_ci_hi": np.exp(log_mean + 1.96 * sigma_total),
                "Ne_estimated": ne_estimated,
                "Ne_used": ne_used,
                "r_per_bp": np.exp(log_mean) / (4.0 * ne_used),
                "r_ci_lo": np.exp(log_mean - 1.96 * sigma_total) / (4.0 * ne_used),
                "r_ci_hi": np.exp(log_mean + 1.96 * sigma_total) / (4.0 * ne_used),
            })
            output["checkpoint_sha256"] = f"ensemble_manifest:{manifest_hash}"
            output["stats_sha256"] = f"ensemble_manifest:{manifest_hash}"
            output["ensemble_members"] = args.members
            output["ensemble_rule"] = "mean_log_rho_with_total_variance"
            np.savez_compressed(args.output_dir / reference_path.name, **output)
        finally:
            for item in loaded:
                item.close()
    print(f"ensembled={len(files)} members={args.members} manifest_sha256={manifest_hash}")


if __name__ == "__main__":
    main()
