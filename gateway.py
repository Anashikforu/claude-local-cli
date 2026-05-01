#!/usr/bin/env python3
"""Anthropic Messages -> MLX-LM OpenAI chat gateway for Claude Code."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_MLX_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "models/Qwen2.5-Coder-7B-Instruct-4bit"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
USER_AGENT = "ClaudeLocalCLI/0.1 (+https://github.com/Anashikforu/claude-local-cli)"


def now_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def estimate_tokens(value: Any) -> int:
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return max(1, len(text) // 4)


def anthropic_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_result":
            result = block.get("content", "")
            if isinstance(result, list):
                result = anthropic_content_to_text(result)
            parts.append(
                f"Tool result for {block.get('tool_use_id', 'unknown')}:\n{result}"
            )
        elif block_type == "image":
            parts.append("[Image input omitted by local text gateway]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n\n".join(part for part in parts if part)


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                if block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", now_id("toolu")),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            converted_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(text_parts) or None,
            }
            if tool_calls:
                converted_message["tool_calls"] = tool_calls
            converted.append(converted_message)
            continue

        if role == "user" and isinstance(content, list):
            tool_result_blocks = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            if tool_result_blocks:
                for block in tool_result_blocks:
                    result_content = block.get("content", "")
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", "unknown"),
                            "content": anthropic_content_to_text(result_content),
                        }
                    )
                remaining = [
                    block
                    for block in content
                    if not (isinstance(block, dict) and block.get("type") == "tool_result")
                ]
                if remaining:
                    converted.append(
                        {"role": "user", "content": anthropic_content_to_text(remaining)}
                    )
                continue

        converted.append({"role": role, "content": anthropic_content_to_text(content)})
    return converted


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
        )
    return converted


def web_tools_enabled() -> bool:
    return os.getenv("ENABLE_WEB_TOOLS", "1").lower() not in ("0", "false", "no")


def accuracy_policy_enabled() -> bool:
    return os.getenv("ENABLE_ACCURACY_POLICY", "1").lower() not in ("0", "false", "no")


def accuracy_policy_prompt() -> str:
    return (
        "Accuracy policy: Be conservative and evidence-based. Do not invent facts, "
        "names, affiliations, dates, URLs, APIs, commands, or citations. If the "
        "answer is not available from the conversation, provided files, or tool "
        "results, say that you do not know. For current, niche, personal, legal, "
        "medical, financial, or source-specific claims, use available web tools "
        "before answering. If search or fetch results are missing, weak, ambiguous, "
        "or contradictory, explain that the evidence is insufficient instead of "
        "guessing. Do not provide suggestions as facts unless they are supported "
        "by evidence; label uncertain reasoning clearly as uncertainty."
    )


def local_web_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Returns a small list of result titles, URLs, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return.",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch a web page URL and return readable text from the page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The http or https URL to fetch.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum number of extracted text characters to return.",
                            "minimum": 500,
                            "maximum": 20000,
                            "default": 6000,
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def add_local_web_tools(openai_request: dict[str, Any]) -> None:
    if not web_tools_enabled():
        return

    tools = openai_request.setdefault("tools", [])
    existing_names = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, dict)
    }
    for tool in local_web_tools():
        name = tool["function"]["name"]
        if name not in existing_names:
            tools.append(tool)

    if "tool_choice" not in openai_request:
        openai_request["tool_choice"] = "auto"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg") and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def http_get_text(url: str, timeout: float, max_bytes: int = 2_000_000) -> str:
    request = urllib.request.Request(url, headers={"user-agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(max_bytes)
    text = raw.decode(charset, errors="replace")
    if "text/html" not in content_type:
        return text
    parser = TextExtractor()
    parser.feed(text)
    return parser.text()


def http_json(
    url: str,
    timeout: float,
    *,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"accept": "application/json", "user-agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    payload = None
    method = "GET"
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        request_headers["content-type"] = "application/json"
        method = "POST"

    request = urllib.request.Request(
        url,
        data=payload,
        headers=request_headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def web_search(query: str, max_results: int, timeout: float) -> dict[str, Any]:
    max_results = max(1, min(max_results, 10))
    provider = os.getenv("WEB_SEARCH_PROVIDER", "auto").lower()

    if provider in ("auto", "tavily") and os.getenv("TAVILY_API_KEY"):
        return tavily_search(query, max_results, timeout)
    if provider in ("auto", "brave") and os.getenv("BRAVE_SEARCH_API_KEY"):
        return brave_search(query, max_results, timeout)
    if provider not in ("auto", "duckduckgo", "ddg"):
        raise ValueError(
            "WEB_SEARCH_PROVIDER must be auto, tavily, brave, or duckduckgo"
        )
    return duckduckgo_search(query, max_results, timeout)


def tavily_search(query: str, max_results: int, timeout: float) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
        "include_answer": True,
        "include_raw_content": False,
    }
    response = http_json(
        TAVILY_SEARCH_URL,
        timeout,
        data=payload,
        headers={"authorization": f"Bearer {api_key}"},
    )
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score"),
            }
        )
    return {
        "provider": "tavily",
        "query": query,
        "answer": response.get("answer", ""),
        "results": results,
    }


def brave_search(query: str, max_results: int, timeout: float) -> dict[str, Any]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    response = http_json(
        f"{BRAVE_SEARCH_URL}?{params}",
        timeout,
        headers={
            "accept": "application/json",
            "x-subscription-token": api_key,
        },
    )
    results = []
    for item in response.get("web", {}).get("results", []):
        snippets = item.get("extra_snippets") or []
        description = item.get("description", "")
        if snippets:
            description = f"{description}\n" + "\n".join(snippets)
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": clean_html_fragment(description),
            }
        )
    return {"provider": "brave", "query": query, "results": results}


def duckduckgo_search(query: str, max_results: int, timeout: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        f"{DUCKDUCKGO_SEARCH_URL}?{params}",
        headers={"user-agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        html_text = response.read(1_000_000).decode("utf-8", errors="replace")

    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>',
        html_text,
        re.DOTALL,
    )
    clean_snippets = [
        clean_html_fragment(first or second)
        for first, second in snippets
    ]

    for index, match in enumerate(pattern.finditer(html_text)):
        href = html.unescape(match.group("href"))
        title = clean_html_fragment(match.group("title"))
        parsed = urllib.parse.urlparse(href)
        query_params = urllib.parse.parse_qs(parsed.query)
        url = query_params.get("uddg", [href])[0]
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": clean_snippets[index] if index < len(clean_snippets) else "",
            }
        )
        if len(results) >= max_results:
            break

    return {"provider": "duckduckgo", "query": query, "results": results}


def clean_html_fragment(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def execute_local_tool(name: str, arguments: Any, timeout: float) -> str:
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("web_search requires a query")
            result = web_search(
                query,
                int(arguments.get("max_results", 5)),
                timeout,
            )
        elif name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError("web_fetch requires an http or https URL")
            max_chars = max(500, min(int(arguments.get("max_chars", 6000)), 20000))
            text = http_get_text(url, timeout)[:max_chars]
            result = {"url": url, "text": text}
        else:
            raise ValueError(f"Unknown local tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool errors should return to the model
        result = {"error": exc.__class__.__name__, "message": str(exc)}
    return json.dumps(result, ensure_ascii=False)


def extract_plain_text_tool_call(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = clean_model_text(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(value, dict):
        return None

    name = value.get("name") or value.get("tool") or value.get("function")
    arguments = value.get("arguments") or value.get("input") or {}
    if name not in ("web_search", "web_fetch"):
        return None
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name, "arguments": arguments}


def parse_tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw_arguments": arguments}


def clean_model_text(text: str, *, strip: bool = True) -> str:
    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(token, "")
    if strip:
        return text.strip()
    return text


def openai_to_anthropic_response(openai_response: dict[str, Any], model: str, request: dict[str, Any]) -> dict[str, Any]:
    choice = (openai_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    content: list[dict[str, Any]] = []

    text = message.get("content")
    if text:
        text = clean_model_text(text)
        content.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", now_id("toolu")),
                "name": function.get("name", ""),
                "input": parse_tool_arguments(function.get("arguments")),
            }
        )

    stop_reason = "tool_use" if any(block["type"] == "tool_use" for block in content) else "end_turn"
    if finish_reason == "length":
        stop_reason = "max_tokens"

    usage = openai_response.get("usage") or {}
    return {
        "id": openai_response.get("id", now_id("msg")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", estimate_tokens(request)),
            "output_tokens": usage.get("completion_tokens", estimate_tokens(content)),
        },
    }


def anthropic_to_openai_request(request: dict[str, Any], default_model: str) -> dict[str, Any]:
    requested_model = request.get("model") or default_model
    model = resolve_model(requested_model, default_model)
    openai_request: dict[str, Any] = {
        "model": model,
        "messages": convert_messages(request.get("messages", [])),
        "stream": bool(request.get("stream", False)),
        "_anthropic_model": requested_model,
    }

    system = request.get("system")
    if system:
        system_text = anthropic_content_to_text(system)
        openai_request["messages"].insert(0, {"role": "system", "content": system_text})

    if accuracy_policy_enabled():
        openai_request["messages"].insert(
            0,
            {"role": "system", "content": accuracy_policy_prompt()},
        )

    if web_tools_enabled():
        openai_request["messages"].insert(
            0,
            {
                "role": "system",
                "content": (
                    "You can use web_search for current web results and web_fetch "
                    "to read a specific URL when live information is needed. Use "
                    "these tools before answering questions that require current "
                    "or source-specific information."
                ),
            },
        )

    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ):
        if src in request:
            openai_request[dst] = request[src]

    tools = convert_tools(request.get("tools"))
    if tools:
        openai_request["tools"] = tools
        tool_choice = request.get("tool_choice")
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "auto":
                openai_request["tool_choice"] = "auto"
            elif tool_choice.get("type") == "any":
                openai_request["tool_choice"] = "required"
            elif tool_choice.get("type") == "tool":
                openai_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")},
                }

    add_local_web_tools(openai_request)
    return openai_request


def resolve_model(requested_model: str, default_model: str) -> str:
    return model_aliases(default_model).get(requested_model, requested_model)


def model_aliases(default_model: str) -> dict[str, str]:
    aliases = {
        "default": default_model,
        "local": default_model,
        "mlx": default_model,
        "gemma": default_model,
        "local-coder": default_model,
        "qwen-coder": default_model,
        "claude-3-5-sonnet": default_model,
        "claude-3-5-haiku": default_model,
        "claude-3-opus": default_model,
        "claude-opus-4-1": default_model,
        "claude-opus-4-5": default_model,
        "claude-sonnet-4": default_model,
        "claude-sonnet-4-5": default_model,
        "claude-sonnet-4-20250514": default_model,
        "claude-haiku-4-5": default_model,
    }
    env_aliases = os.getenv("MODEL_ALIASES", "")
    for item in env_aliases.split(","):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        aliases[key.strip()] = value.strip()
    return aliases


def available_model_ids(default_model: str) -> list[str]:
    model_ids = [
        os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
        "local-coder",
        "qwen-coder",
        default_model,
    ]
    model_ids.extend(model_aliases(default_model).keys())
    return list(dict.fromkeys(model_id for model_id in model_ids if model_id))


def is_available_model(model_id: str, default_model: str) -> bool:
    return model_id in available_model_ids(default_model) or resolve_model(
        model_id, default_model
    ) == default_model


def model_object(model_id: str) -> dict[str, Any]:
    return {
        "type": "model",
        "id": model_id,
        "display_name": model_id,
        "created_at": "2026-05-01T00:00:00Z",
    }


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "ClaudeMLXGateway/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8") or "{}")

    def write_json(self, status: int, value: dict[str, Any]) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def route_path(self) -> str:
        return urllib.parse.urlsplit(self.path).path

    def do_HEAD(self) -> None:
        path = self.route_path()
        if path in ("/health", "/"):
            self.send_response(200)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.route_path()
        if path in ("/health", "/"):
            self.write_json(
                200,
                {
                    "ok": True,
                    "mlx_base_url": self.server.mlx_base_url,  # type: ignore[attr-defined]
                    "default_model": self.server.default_model,  # type: ignore[attr-defined]
                },
            )
            return
        if path == "/v1/models":
            model_ids = available_model_ids(self.server.default_model)  # type: ignore[attr-defined]
            self.write_json(
                200,
                {
                    "data": [model_object(model_id) for model_id in model_ids],
                    "has_more": False,
                    "first_id": model_ids[0],
                    "last_id": model_ids[-1],
                },
            )
            return
        if path.startswith("/v1/models/"):
            model_id = path.removeprefix("/v1/models/")
            if is_available_model(model_id, self.server.default_model):  # type: ignore[attr-defined]
                self.write_json(200, model_object(model_id))
                return
            self.write_json(
                404,
                {
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": f"Model {model_id} not found",
                    },
                },
            )
            return
        self.write_json(404, {"error": {"type": "not_found", "message": path}})

    def do_POST(self) -> None:
        path = self.route_path()
        if path == "/v1/messages/count_tokens":
            request = self.read_json()
            self.write_json(200, {"input_tokens": estimate_tokens(request)})
            return

        if path != "/v1/messages":
            self.write_json(404, {"error": {"type": "not_found", "message": path}})
            return

        try:
            request = self.read_json()
            openai_request = anthropic_to_openai_request(
                request, self.server.default_model  # type: ignore[attr-defined]
            )
            if openai_request.get("stream"):
                self.proxy_stream(request, openai_request)
            else:
                self.proxy_json(request, openai_request)
        except Exception as exc:  # noqa: BLE001 - this is an HTTP boundary
            self.write_json(
                500,
                {
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                },
            )

    def proxy_json(self, request: dict[str, Any], openai_request: dict[str, Any]) -> None:
        try:
            payload = self.complete_with_local_tools(openai_request)
        except urllib.error.HTTPError as exc:
            self.write_json(exc.code, {"error": exc.read().decode("utf-8")})
            return

        self.write_json(
            200,
            openai_to_anthropic_response(
                payload,
                openai_request.get("_anthropic_model") or openai_request["model"],
                request,
            ),
        )

    def openai_chat_completion(self, openai_request: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.server.mlx_base_url}/v1/chat/completions"  # type: ignore[attr-defined]
        upstream_request = {
            key: value for key, value in openai_request.items() if not key.startswith("_")
        }
        upstream = urllib.request.Request(
            url,
            data=json.dumps(upstream_request).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(upstream, timeout=self.server.timeout) as response:  # type: ignore[attr-defined]
            return json.loads(response.read().decode("utf-8"))

    def complete_with_local_tools(self, openai_request: dict[str, Any]) -> dict[str, Any]:
        request_for_loop = dict(openai_request)
        request_for_loop["stream"] = False

        for _ in range(3):
            payload = self.openai_chat_completion(request_for_loop)
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            plain_tool_call = extract_plain_text_tool_call(message.get("content"))
            local_tool_calls = [
                tool_call
                for tool_call in tool_calls
                if (tool_call.get("function") or {}).get("name")
                in ("web_search", "web_fetch")
            ]
            if not local_tool_calls and not plain_tool_call:
                return payload
            if tool_calls and len(local_tool_calls) != len(tool_calls):
                return payload

            if plain_tool_call:
                tool_result = execute_local_tool(
                    plain_tool_call["name"],
                    plain_tool_call["arguments"],
                    self.server.timeout,  # type: ignore[attr-defined]
                )
                request_for_loop["messages"].append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                    }
                )
                request_for_loop["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool result for {plain_tool_call['name']}:\n"
                            f"{tool_result}\n\n"
                            "Use these results to answer the original question. "
                            "Do not output another tool JSON object unless another "
                            "web_search or web_fetch is strictly necessary."
                        ),
                    }
                )
                continue

            request_for_loop["messages"].append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            for tool_call in local_tool_calls:
                function = tool_call.get("function") or {}
                request_for_loop["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", now_id("toolu")),
                        "content": execute_local_tool(
                            function.get("name", ""),
                            parse_tool_arguments(function.get("arguments")),
                            self.server.timeout,  # type: ignore[attr-defined]
                        ),
                    }
                )

        return payload

    def proxy_stream(self, request: dict[str, Any], openai_request: dict[str, Any]) -> None:
        if web_tools_enabled():
            try:
                payload = self.complete_with_local_tools(openai_request)
            except urllib.error.HTTPError as exc:
                self.send_json_stream_error(exc.read().decode("utf-8"))
                return
            self.send_anthropic_text_stream(request, openai_request, payload)
            return

        url = f"{self.server.mlx_base_url}/v1/chat/completions"  # type: ignore[attr-defined]
        upstream_request = {
            key: value for key, value in openai_request.items() if not key.startswith("_")
        }
        upstream = urllib.request.Request(
            url,
            data=json.dumps(upstream_request).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        message_id = now_id("msg")
        model = openai_request.get("_anthropic_model") or openai_request["model"]
        output_tokens = 0
        self.send_event("message_start", {"type": "message_start", "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": estimate_tokens(request), "output_tokens": 0},
        }})
        self.send_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})

        try:
            with urllib.request.urlopen(upstream, timeout=self.server.timeout) as response:  # type: ignore[attr-defined]
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        text = clean_model_text(text, strip=False)
                    if text:
                        output_tokens += estimate_tokens(text)
                        self.send_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
        except urllib.error.HTTPError as exc:
            self.send_event("error", {"type": "error", "error": {"type": "upstream_error", "message": exc.read().decode("utf-8")}})
            return

        self.send_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        self.send_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": output_tokens}})
        self.send_event("message_stop", {"type": "message_stop"})

    def send_json_stream_error(self, message: str) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        self.send_event(
            "error",
            {
                "type": "error",
                "error": {"type": "upstream_error", "message": message},
            },
        )

    def send_anthropic_text_stream(
        self,
        request: dict[str, Any],
        openai_request: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        model = openai_request.get("_anthropic_model") or openai_request["model"]
        anthropic_response = openai_to_anthropic_response(payload, model, request)
        text = "".join(
            block.get("text", "")
            for block in anthropic_response.get("content", [])
            if block.get("type") == "text"
        )

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        self.send_event("message_start", {"type": "message_start", "message": {
            "id": anthropic_response["id"],
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": estimate_tokens(request), "output_tokens": 0},
        }})
        self.send_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        if text:
            self.send_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
        self.send_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        self.send_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": anthropic_response["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": estimate_tokens(text)}})
        self.send_event("message_stop", {"type": "message_stop"})

    def send_event(self, event: str, payload: dict[str, Any]) -> None:
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code to MLX-LM gateway")
    parser.add_argument("--host", default=os.getenv("GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GATEWAY_PORT", "3001")))
    parser.add_argument("--mlx-base-url", default=os.getenv("MLX_BASE_URL", DEFAULT_MLX_BASE_URL))
    parser.add_argument("--default-model", default=os.getenv("MLX_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("MLX_TIMEOUT", "600")))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), GatewayHandler)
    server.mlx_base_url = args.mlx_base_url.rstrip("/")  # type: ignore[attr-defined]
    server.default_model = args.default_model  # type: ignore[attr-defined]
    server.timeout = args.timeout  # type: ignore[attr-defined]
    server.verbose = args.verbose  # type: ignore[attr-defined]

    print(
        f"Claude MLX gateway listening on http://{args.host}:{args.port} "
        f"-> {server.mlx_base_url}"
    )
    print(f"Default model: {server.default_model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
