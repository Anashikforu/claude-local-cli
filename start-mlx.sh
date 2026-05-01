#!/usr/bin/env bash
set -euo pipefail

: "${MLX_MODEL:=models/Qwen2.5-Coder-7B-Instruct-4bit}"
: "${MLX_PORT:=8080}"

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv. Create it with:" >&2
  echo "  uv venv --python /opt/homebrew/bin/python3.11 .venv" >&2
  echo "  uv pip install mlx-lm" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

exec .venv/bin/python -m mlx_lm.server \
  --model "$MLX_MODEL" \
  --port "$MLX_PORT"
