#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

: "${MLX_MODEL:=models/Qwen2.5-Coder-7B-Instruct-4bit}"
: "${MLX_PORT:=8080}"
: "${GATEWAY_PORT:=3001}"
: "${MLX_BASE_URL:=http://127.0.0.1:${MLX_PORT}}"
: "${CLAUDE_MODEL:=claude-sonnet-4-5}"
: "${WEB_POLICY:=auto}"
: "${SEARCH_DEPTH:=agentic}"
: "${WEB_CONTEXT_SIZE:=medium}"
: "${WEB_MAX_USES:=2}"
: "${WEB_MAX_TOOL_RESULT_CHARS:=6000}"
: "${AUTO_WEB_MAX_CHARS:=1500}"
: "${WEB_USE_JINA_READER:=1}"
: "${TAVILY_INCLUDE_RAW_CONTENT:=0}"
: "${WEB_BLOCKED_DOMAINS:=reddit.com,quora.com}"
: "${AUTO_WEB_TIMEOUT:=8}"
: "${ENABLE_EVAL_LOGGING:=1}"
: "${EVAL_LOG_PATH:=logs/gateway.jsonl}"
: "${EVAL_LOG_TEXT:=0}"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv. Create it with:" >&2
  echo "  uv venv --python /opt/homebrew/bin/python3.11 .venv" >&2
  echo "  uv pip install mlx-lm" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "Could not find the claude CLI in PATH." >&2
  exit 1
fi

MLX_LOG="${MLX_LOG:-/tmp/claude-mlx-server.log}"
GATEWAY_LOG="${GATEWAY_LOG:-/tmp/claude-mlx-gateway.log}"
MLX_PID=""
GATEWAY_PID=""

cleanup() {
  if [ -n "$GATEWAY_PID" ] && kill -0 "$GATEWAY_PID" 2>/dev/null; then
    kill "$GATEWAY_PID" 2>/dev/null || true
  fi
  if [ -n "$MLX_PID" ] && kill -0 "$MLX_PID" 2>/dev/null; then
    kill "$MLX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

stop_existing_listener() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  local pids
  pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return 0
  fi

  echo "Stopping existing listener on port ${port}: ${pids}"
  kill $pids 2>/dev/null || true
  sleep 1
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local attempts="${3:-120}"

  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for ${label}: ${url}" >&2
  return 1
}

echo "Starting MLX-LM on ${MLX_BASE_URL}"
stop_existing_listener "$MLX_PORT"
stop_existing_listener "$GATEWAY_PORT"

if [ ! -e "$MLX_MODEL" ] && [[ "$MLX_MODEL" == models/* ]]; then
  echo "Model path does not exist: ${MLX_MODEL}" >&2
  echo "Run ./download-coder-model.sh first." >&2
  exit 1
fi

MLX_MODEL="$MLX_MODEL" MLX_PORT="$MLX_PORT" ./start-mlx.sh >"$MLX_LOG" 2>&1 &
MLX_PID="$!"

wait_for_url "MLX-LM" "${MLX_BASE_URL}/v1/models" 180 || {
  echo "MLX-LM log:" >&2
  tail -80 "$MLX_LOG" >&2 || true
  exit 1
}

echo "Starting Claude MLX gateway on http://127.0.0.1:${GATEWAY_PORT}"
MLX_MODEL="$MLX_MODEL" \
MLX_BASE_URL="$MLX_BASE_URL" \
GATEWAY_PORT="$GATEWAY_PORT" \
WEB_POLICY="$WEB_POLICY" \
SEARCH_DEPTH="$SEARCH_DEPTH" \
WEB_CONTEXT_SIZE="$WEB_CONTEXT_SIZE" \
WEB_MAX_USES="$WEB_MAX_USES" \
WEB_MAX_TOOL_RESULT_CHARS="$WEB_MAX_TOOL_RESULT_CHARS" \
AUTO_WEB_MAX_CHARS="$AUTO_WEB_MAX_CHARS" \
WEB_USE_JINA_READER="$WEB_USE_JINA_READER" \
TAVILY_INCLUDE_RAW_CONTENT="$TAVILY_INCLUDE_RAW_CONTENT" \
WEB_BLOCKED_DOMAINS="$WEB_BLOCKED_DOMAINS" \
WEB_ALLOWED_DOMAINS="${WEB_ALLOWED_DOMAINS:-}" \
WEB_SEARCH_PROVIDER="${WEB_SEARCH_PROVIDER:-auto}" \
TAVILY_API_KEY="${TAVILY_API_KEY:-}" \
BRAVE_SEARCH_API_KEY="${BRAVE_SEARCH_API_KEY:-}" \
AUTO_WEB_TIMEOUT="$AUTO_WEB_TIMEOUT" \
ENABLE_EVAL_LOGGING="$ENABLE_EVAL_LOGGING" \
EVAL_LOG_PATH="$EVAL_LOG_PATH" \
EVAL_LOG_TEXT="$EVAL_LOG_TEXT" \
./run-gateway.sh >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID="$!"

wait_for_url "gateway" "http://127.0.0.1:${GATEWAY_PORT}/health" 30 || {
  echo "Gateway log:" >&2
  tail -80 "$GATEWAY_LOG" >&2 || true
  exit 1
}

echo "Testing gateway request path"
if ! curl -fsS --max-time 120 "http://127.0.0.1:${GATEWAY_PORT}/v1/messages" \
  -H 'content-type: application/json' \
  --data "{\"model\":\"${CLAUDE_MODEL}\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"Say ok\"}]}" >/tmp/claude-mlx-self-test.json; then
  echo "Gateway self-test failed." >&2
  echo "Gateway log:" >&2
  tail -120 "$GATEWAY_LOG" >&2 || true
  echo "MLX log:" >&2
  tail -120 "$MLX_LOG" >&2 || true
  exit 1
fi
echo "Gateway self-test passed"

echo "Starting Claude Code with local model: ${CLAUDE_MODEL}"
echo "Web policy: ${WEB_POLICY}, search depth: ${SEARCH_DEPTH}, context: ${WEB_CONTEXT_SIZE}, max uses: ${WEB_MAX_USES}"
echo "MLX log: ${MLX_LOG}"
echo "Gateway log: ${GATEWAY_LOG}"

if [ "${1:-}" = "--debug-services" ]; then
  echo "Services are running. Test Claude manually with:"
  echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:${GATEWAY_PORT} ANTHROPIC_AUTH_TOKEN=local-mlx claude --bare --model ${CLAUDE_MODEL}"
  echo "Press Ctrl+C here to stop MLX-LM and the gateway."
  while true; do
    sleep 3600
  done
fi

ANTHROPIC_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}" \
ANTHROPIC_AUTH_TOKEN="local-mlx" \
ANTHROPIC_API_KEY="local-mlx" \
ANTHROPIC_MODEL="$CLAUDE_MODEL" \
ANTHROPIC_DEFAULT_OPUS_MODEL="$CLAUDE_MODEL" \
ANTHROPIC_DEFAULT_SONNET_MODEL="$CLAUDE_MODEL" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="$CLAUDE_MODEL" \
CLAUDE_CODE_SUBAGENT_MODEL="$CLAUDE_MODEL" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude --bare --model "$CLAUDE_MODEL"
