#!/usr/bin/env bash
set -euo pipefail

export MODEL_REPO="${MODEL_REPO:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
export MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-Coder-7B-Instruct-4bit}"

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv. Create it with:" >&2
  echo "  uv venv --python /opt/homebrew/bin/python3.11 .venv" >&2
  echo "  uv pip install mlx-lm" >&2
  exit 1
fi

mkdir -p "$(dirname "$MODEL_DIR")"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "Downloading ${MODEL_REPO}"
echo "Destination: ${MODEL_DIR}"

.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)
PY

echo "Downloaded model to ${MODEL_DIR}"
echo "Test from normal macOS Terminal:"
echo "  .venv/bin/python -m mlx_lm.generate --model ${MODEL_DIR} --prompt 'Write a Python function to add two numbers.'"
