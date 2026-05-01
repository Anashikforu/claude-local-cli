# Claude Local CLI

Run Claude Code against local MLX models through a lightweight Anthropic-compatible gateway.

Claude Code expects an Anthropic-compatible API:

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

`mlx_lm.server` exposes an OpenAI-compatible API:

- `POST /v1/chat/completions`

This gateway translates between the two.

## Web Tools

Claude Local CLI adds two gateway-side tools for the local model:

- `web_search(query, max_results)` searches the web for current results.
- `web_fetch(url, max_chars)` fetches a URL and returns readable page text.

The model still runs locally, but these tools use your normal internet connection when the model requests live information. Disable them with:

```bash
ENABLE_WEB_TOOLS=0 ./start-local-claude.sh
```

## One-command Start

Run everything together:

```bash
cd /Users/ashiks/Downloads/claude-local-cli
./download-coder-model.sh
./start-local-claude.sh
```

This starts MLX-LM, starts the gateway, then opens Claude Code with the right environment variables. When Claude Code exits, the script stops the local background services.

The launcher starts Claude Code in `--bare` mode. The gateway maps the Claude Code-facing model ID to the downloaded MLX model before sending the request to MLX-LM.

Optional overrides:

```bash
MLX_MODEL=models/Qwen2.5-Coder-7B-Instruct-4bit \
CLAUDE_MODEL=qwen-coder \
./start-local-claude.sh
```

## Manual Start

### 1. Start MLX-LM

This repository has a local `.venv` configured for MLX-LM. If you need to recreate it:

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
uv pip install mlx-lm
```

Start your model:

```bash
cd /Users/ashiks/Downloads/claude-local-cli
./download-coder-model.sh
./start-mlx.sh
```

Use a model and context length that fit your Mac. Claude Code usually needs a large context window.

### 2. Start the gateway

In another terminal:

```bash
cd /Users/ashiks/Downloads/claude-local-cli
./run-gateway.sh
```

Optional overrides:

```bash
MLX_MODEL=models/Qwen2.5-Coder-7B-Instruct-4bit \
MLX_BASE_URL=http://127.0.0.1:8080 \
GATEWAY_PORT=3001 \
./run-gateway.sh
```

### 3. Start Claude Code through the gateway

In a third terminal:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:3001
export ANTHROPIC_AUTH_TOKEN=local-mlx

claude --bare --model qwen-coder
```

## Notes

- Non-streaming responses support text and tool calls.
- Streaming currently streams text deltas. For heavy Claude Code tool workflows, use non-streaming if the local model/runtime has trouble emitting tool calls.
- Local models vary a lot in tool-calling quality. If Gemma fails to edit files reliably, try a coding/tool model in MLX format.
- MLX requires access to your Mac's Metal device. It may fail inside sandboxed/headless agent sessions, but it should run from the normal macOS Terminal app.
