#!/usr/bin/env bash
# Redirect to project scripts/run_ablation_temp_ntk.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec bash "$ROOT/scripts/run_ablation_temp_ntk.sh" "$@"
