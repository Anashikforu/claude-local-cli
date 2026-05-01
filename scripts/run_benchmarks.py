#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, data: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def response_text(response: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "")
        for block in response.get("content", [])
        if block.get("type") == "text"
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def run_case(base_url: str, model: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.time()
    response = post_json(
        f"{base_url}/v1/messages",
        {
            "model": model,
            "max_tokens": case.get("max_tokens", 320),
            "messages": [{"role": "user", "content": case["prompt"]}],
        },
        timeout,
    )
    latency = time.time() - started
    text = response_text(response)

    failures = []
    if not text.strip():
        failures.append("empty_output")
    for needle in case.get("must_contain", []):
        if needle.lower() not in text.lower():
            failures.append(f"missing:{needle}")
    if case.get("must_contain_any") and not contains_any(text, case["must_contain_any"]):
        failures.append("missing_any:" + "|".join(case["must_contain_any"]))
    for needle in case.get("must_not_contain", []):
        if needle.lower() in text.lower():
            failures.append(f"forbidden:{needle}")
    max_latency = float(case.get("max_latency_seconds", timeout))
    if latency > max_latency:
        failures.append(f"latency>{max_latency}")

    return {
        "name": case["name"],
        "category": case.get("category", ""),
        "passed": not failures,
        "failures": failures,
        "latency_seconds": round(latency, 2),
        "text_preview": text[:500],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claude Local CLI response benchmarks")
    parser.add_argument("--base-url", default="http://127.0.0.1:3001")
    parser.add_argument("--model", default="qwen-coder")
    parser.add_argument("--cases", default="benchmarks/prompts.jsonl")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--json", action="store_true", help="Print JSONL results only")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    cases = load_cases(Path(args.cases))
    results = []

    for case in cases:
        try:
            result = run_case(base_url, args.model, case, args.timeout)
        except urllib.error.HTTPError as exc:
            result = {
                "name": case["name"],
                "category": case.get("category", ""),
                "passed": False,
                "failures": [f"http_{exc.code}"],
                "latency_seconds": 0,
                "text_preview": exc.read().decode("utf-8", errors="replace")[:500],
            }
        except Exception as exc:
            result = {
                "name": case["name"],
                "category": case.get("category", ""),
                "passed": False,
                "failures": [f"{exc.__class__.__name__}: {exc}"],
                "latency_seconds": 0,
                "text_preview": "",
            }
        results.append(result)
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} {result['name']} {result['latency_seconds']}s")
            if result["failures"]:
                print("  failures:", ", ".join(result["failures"]))
            print("  preview:", result["text_preview"].replace("\n", " ")[:240])

    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    if not args.json:
        print(f"\nSummary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
