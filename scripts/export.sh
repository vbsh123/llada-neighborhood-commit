#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
experiment_path="${1:?Usage: bash scripts/export.sh runs/RUN_NAME}"
[[ -d "$experiment_path" ]] || { echo "Missing run directory: $experiment_path"; exit 1; }
mkdir -p exports
experiment_base="$(basename "$experiment_path")"
archive_path="exports/${experiment_base}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
# No model weights or HF credentials: archive only the requested run directory.
tar --exclude='*.partial' --exclude='*.tmp' --exclude='.lock' \
  -czf "$archive_path" -C "$(dirname "$experiment_path")" "$experiment_base"
sha256sum "$archive_path" > "$archive_path.sha256"
echo "Export: $archive_path"
echo "Checksum: $archive_path.sha256"
