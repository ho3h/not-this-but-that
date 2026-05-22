#!/usr/bin/env bash
# Run remaining gauntlet attacks sequentially.
# Usage: bash scripts/gauntlet/run_chain.sh A3 A4 A5 A6 A7

set -e
cd "$(dirname "$0")/../.."

LOG_DIR="logs/gauntlet"
mkdir -p "$LOG_DIR"

for attack in "$@"; do
    lc=$(echo "$attack" | tr '[:upper:]' '[:lower:]')
    case "$lc" in
        a3) script="scripts.gauntlet.a3_few_shot" ;;
        a4) script="scripts.gauntlet.a4_scalpel_mid" ;;
        a5) script="scripts.gauntlet.a5_scalpel_pre" ;;
        a6) script="scripts.gauntlet.a6_orthogonalize" ;;
        a7) script="scripts.gauntlet.a7_caa" ;;
        *) echo "unknown attack: $attack"; exit 1 ;;
    esac
    echo "=== $(date +%H:%M:%S) running $attack ($script) ==="
    uv run python -m "$script" 2>&1 | tee "$LOG_DIR/${lc}.log"
    echo "=== $(date +%H:%M:%S) $attack complete ==="
done

echo "=== chain finished ==="
