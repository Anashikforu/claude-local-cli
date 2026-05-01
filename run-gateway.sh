#!/usr/bin/env bash
set -euo pipefail

: "${MLX_BASE_URL:=http://127.0.0.1:8080}"
: "${MLX_MODEL:=models/Qwen2.5-Coder-7B-Instruct-4bit}"
: "${GATEWAY_PORT:=3001}"

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

exec "$PYTHON" -u gateway.py --verbose \
  --port "$GATEWAY_PORT" \
  --mlx-base-url "$MLX_BASE_URL" \
  --default-model "$MLX_MODEL"
