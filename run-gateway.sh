#!/usr/bin/env bash
set -euo pipefail

: "${MLX_BASE_URL:=http://127.0.0.1:8080}"
: "${MLX_MODEL:=models/Qwen2.5-Coder-7B-Instruct-4bit}"
: "${GATEWAY_PORT:=3001}"
: "${WEB_POLICY:=auto}"
: "${SEARCH_DEPTH:=agentic}"
: "${WEB_CONTEXT_SIZE:=medium}"
: "${WEB_MAX_USES:=2}"
: "${WEB_USE_JINA_READER:=1}"
: "${WEB_BLOCKED_DOMAINS:=reddit.com,quora.com}"
: "${AUTO_WEB_TIMEOUT:=8}"

export WEB_POLICY SEARCH_DEPTH WEB_CONTEXT_SIZE WEB_MAX_USES WEB_USE_JINA_READER
export WEB_BLOCKED_DOMAINS AUTO_WEB_TIMEOUT

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

exec "$PYTHON" -u gateway.py --verbose \
  --port "$GATEWAY_PORT" \
  --mlx-base-url "$MLX_BASE_URL" \
  --default-model "$MLX_MODEL"
