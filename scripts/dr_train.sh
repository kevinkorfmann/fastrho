#!/usr/bin/env bash
# Compatibility entry point for the primary domain-randomized model.
# This script submits the complete numbered Slurm workflow; it performs no computation itself.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${REPO}/models/domain-randomized-v1/reproduce/submit.sh" "$@"
