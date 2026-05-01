#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def request_json(url: str, *, method: str = "GET", data: dict[str, Any] | None = None, timeout: float = 120) -> dict[str, Any]:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"content-type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def message_text(response: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "")
        for block in response.get("content", [])
        if block.get("type") == "text"
    )


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{status} {name}{': ' + detail if detail else ''}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running Claude Local CLI gateway")
    parser.add_argument("--base-url", default="http://127.0.0.1:3001")
    parser.add_argument("--model", default="qwen-coder")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    failures = 0

    try:
        health = request_json(f"{base_url}/health", timeout=args.timeout)
        failures += not check("health", health.get("ok") is True, json.dumps(health))

        models = request_json(f"{base_url}/v1/models", timeout=args.timeout)
        model_ids = [item.get("id") for item in models.get("data", [])]
        failures += not check("models includes qwen-coder", "qwen-coder" in model_ids)

        detail = request_json(f"{base_url}/v1/models/{args.model}", timeout=args.timeout)
        failures += not check("model detail", detail.get("id") == args.model)

        tokens = request_json(
            f"{base_url}/v1/messages/count_tokens",
            method="POST",
            data={"model": args.model, "messages": [{"role": "user", "content": "hello"}]},
            timeout=args.timeout,
        )
        failures += not check("count_tokens", isinstance(tokens.get("input_tokens"), int))

        start = time.time()
        response = request_json(
            f"{base_url}/v1/messages?beta=true",
            method="POST",
            data={
                "model": args.model,
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            },
            timeout=args.timeout,
        )
        text = message_text(response)
        failures += not check("messages beta route", bool(text), text[:120])
        failures += not check("no qwen special token leak", "<|im_" not in text and "<|endoftext|>" not in text)
        print(f"INFO latency_seconds={time.time() - start:.2f}")

    except urllib.error.HTTPError as exc:
        print(f"FAIL http_error status={exc.code} body={exc.read().decode('utf-8', errors='replace')}")
        return 1
    except Exception as exc:
        print(f"FAIL exception {exc.__class__.__name__}: {exc}")
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
