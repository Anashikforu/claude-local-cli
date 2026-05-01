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

For better search quality, configure a real search API. Claude Local CLI uses providers in this order:

1. Tavily when `TAVILY_API_KEY` is set.
2. Brave Search when `BRAVE_SEARCH_API_KEY` is set.
3. DuckDuckGo HTML as a no-key fallback.

Examples:

```bash
TAVILY_API_KEY=tvly-your-key ./start-local-claude.sh
```

```bash
BRAVE_SEARCH_API_KEY=your-key ./start-local-claude.sh
```

Force a provider with `WEB_SEARCH_PROVIDER=auto`, `tavily`, `brave`, or `duckduckgo`.

## Accuracy Policy

Claude Local CLI injects an accuracy policy by default. The model is instructed not to invent facts, citations, commands, URLs, names, affiliations, or dates. When evidence is missing or weak, it should say it does not know instead of guessing.

Automatic web verification uses `smart` mode by default. For normal coding tasks, the gateway keeps the model focused on local files/context. For questions that look current, source-specific, personal, high-risk, docs-related, or explicitly ask to search/verify, the gateway searches before the local model answers and injects the evidence into the prompt.

Tune it with:

```bash
AUTO_WEB_VERIFY=smart \
AUTO_WEB_MODE=balanced \
AUTO_WEB_MAX_RESULTS=5 \
AUTO_WEB_TOOL_BUDGET=2 \
AUTO_WEB_TIMEOUT=8 \
./start-local-claude.sh
```

`AUTO_WEB_VERIFY` accepts:

- `smart`: default; search only when the turn appears to need external/current evidence.
- `ask`: search only when the user explicitly asks to search/verify/use docs/latest.
- `always`: search every turn unless the user says not to.
- `off`: disable automatic verification.

Automatic verification modes:

- `AUTO_WEB_MODE=fast`: search snippets only.
- `AUTO_WEB_MODE=balanced`: search plus one prioritized page fetch. This is the default.
- `AUTO_WEB_MODE=deep`: search plus up to three prioritized page fetches.

Override fetch behavior directly with `AUTO_WEB_FETCH_RESULTS` and fetched text size with `AUTO_WEB_MAX_CHARS`.

The gateway follows a Codex/Claude-style strategy:

- Local codebase work uses local files/context first.
- External, current, source-specific, docs, package, URL, and identity questions trigger web verification.
- User overrides are respected: saying "search", "verify", "latest", or "docs" forces web use; saying "don't search" or "without web" disables it for that request.
- Tool use is budgeted per turn with `AUTO_WEB_TOOL_BUDGET`.

Disable automatic web verification with:

```bash
AUTO_WEB_VERIFY=0 ./start-local-claude.sh
```

Disable this policy with:

```bash
ENABLE_ACCURACY_POLICY=0 ./start-local-claude.sh
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

## Testing and Benchmarks

Run deterministic unit tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Start services without opening an interactive Claude session:

```bash
./start-local-claude.sh --debug-services
```

In another terminal, run gateway smoke tests:

```bash
.venv/bin/python scripts/run_smoke_tests.py --base-url http://127.0.0.1:3001
```

Run response benchmarks:

```bash
.venv/bin/python scripts/run_benchmarks.py --base-url http://127.0.0.1:3001
```

Benchmark cases live in `benchmarks/prompts.jsonl`. Each case can check latency, required text, forbidden text, and special-token leaks.

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
