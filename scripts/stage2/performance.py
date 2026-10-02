#!/usr/bin/env python3
"""Public performance test: streaming requests, timed inside the Arena job.

Shares the prepare, bundle and remote flow of `accuracy.py`; this file adds the input
selection, the streaming measurement of one request, and the summary metrics.
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request

import accuracy

PREFIX = "STAGE2_PERFORMANCE"
INPUTS = ("256", "1024", "2048", "image", "audio")


def read_media_inputs() -> dict[str, dict]:
    """Load `public-test/performance.jsonl`: the image and audio requests, padded with text
    to 512 and 768 prompt tokens as in grading, keyed by dataset."""
    base = accuracy.CRATE / "public-test"
    rows = [json.loads(line) for line in (base / "performance.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        accuracy.expand(row["request"], base)  # Only checks the media; the job expands it again.
    return {row["dataset"]: row for row in rows}


def read_input(name: str) -> dict:
    """The request of one input: a text prompt of that many tokens, or padded public media."""
    if name in ("256", "1024", "2048"):
        source = json.loads((accuracy.HERE / "performance_text.json").read_text())
        prompt = source["256"]
        if name in ("1024", "2048"):
            prompt += source["256_1024"]
        if name == "2048":
            prompt += source["1024_2048"]
        row = {
            "id": f"text.{name}", "suite": name,
            "request": {"model": "gemma-4-12b-it", "messages": [
                {"role": "user", "content": prompt}
            ]},
        }
    elif name in ("image", "audio"):
        source = read_media_inputs()[{"image": "vqav2", "audio": "fleurs_en"}[name]]
        row = {"id": source["id"], "suite": name, "request": source["request"]}
    else:
        raise ValueError("Choose one test: " + ", ".join(INPUTS))
    row["request"].update(max_completion_tokens=64, temperature=0.0, ignore_eos=True,
                          stream=True, stream_options={"include_usage": True})
    return row


def select_inputs(name: str | None, concurrency: int) -> tuple[str, list[dict], list[int]]:
    """Without `--test`, the five inputs in turn; with it, that input alone.

    Both send `ROUNDS * concurrency` requests.
    """
    total = accuracy.ROUNDS * concurrency
    if name is not None:
        return name, [read_input(name)], [0] * total
    return "mixed", [read_input(name) for name in INPUTS], [index % len(INPUTS) for index in range(total)]


def stream_response(response):
    """Yield SSE data events, including events split over multiple data lines."""
    lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))
    if lines:
        yield "\n".join(lines)


def measure_request(row: dict, body: dict, request_index: int, timeout: float,
                    start: threading.Event) -> dict:
    """Send one streaming request and time it. Never raises: a failure is a failed record.

    TTFT runs from just before the request is sent to the first non-empty content delta.
    """
    req = urllib.request.Request(
        accuracy.BASE_URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start.wait()
    dispatched = time.monotonic()
    record = dict(id=row["id"], suite=row["suite"], request_index=request_index, passed=False)
    text, first_text, usage, done = [], None, None, False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("Expected a streaming text/event-stream response")
            for data in stream_response(response):
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(f"Server error: {event['error']}")
                if event.get("usage") is not None:
                    if usage is not None and usage != event["usage"]:
                        raise ValueError("Inconsistent streaming usage")
                    usage = event["usage"]
                choices = event["choices"]
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("Expected at most one streaming choice")
                if choices:
                    content = choices[0]["delta"].get("content")
                    if content is not None and not isinstance(content, str):
                        raise ValueError("Non-text content delta")
                    if content:
                        if first_text is None:
                            first_text = time.monotonic()
                        text.append(content)
                    if choices[0].get("finish_reason") is not None:
                        record["finish_reason"] = choices[0]["finish_reason"]
            completed = time.monotonic()
        if not done or first_text is None or usage is None:
            raise ValueError("Incomplete stream: require text, usage, and [DONE]")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError(f"Invalid usage.{key}")
        tokens = usage["completion_tokens"]
        if tokens == 0 or usage["total_tokens"] != usage["prompt_tokens"] + tokens:
            raise ValueError("Inconsistent token counts")
        if row["suite"] in ("256", "1024", "2048") and usage["prompt_tokens"] != int(row["suite"]):
            raise ValueError(f"Expected {row['suite']} prompt tokens; got {usage['prompt_tokens']}")
        ttft = (first_text - dispatched) * 1000
        latency = (completed - dispatched) * 1000
        record.update(
            passed=True, output="".join(text), usage=usage,
            ttft_ms=ttft, latency_ms=latency,
            estimated_tpot_ms=(latency - ttft) / (tokens - 1) if tokens > 1 else None,
            dispatch_offset_ms=(dispatched - start.epoch) * 1000,
            completion_offset_ms=(completed - start.epoch) * 1000,
        )
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
    record["seconds"] = round(time.monotonic() - dispatched, 3)
    return record


def percentile(values: list, percent: int) -> float | None:
    """Nearest-rank percentile, rounded up; None values are ignored."""
    values = sorted(value for value in values if value is not None)
    return values[max(0, math.ceil(len(values) * percent / 100) - 1)] if values else None


def percentiles(records: list[dict], names: tuple[str, ...]) -> dict:
    return {name: {f"p{p}": percentile([r[name] for r in records], p) for p in (50, 90)}
            for name in names}


def summarize(records: list[dict]) -> dict:
    """p50/p90 of the per-request times, and throughput over the whole run.

    A mixed run also lists TTFT and TPOT for each input.
    """
    # A run with a failed request must not look like a successful measurement.
    if not records or not all(record["passed"] for record in records):
        return {"metrics": None}
    window = (max(r["completion_offset_ms"] for r in records)
              - min(r["dispatch_offset_ms"] for r in records)) / 1000
    metrics = percentiles(records, ("ttft_ms", "latency_ms", "estimated_tpot_ms"))
    metrics.update(
        generated_tokens_per_second=sum(r["usage"]["completion_tokens"] for r in records) / window,
        requests_per_second=len(records) / window,
        measurement_window_seconds=window,
    )
    suites = [name for name in INPUTS if any(r["suite"] == name for r in records)]
    if len(suites) > 1:
        metrics["inputs"] = {
            name: percentiles([r for r in records if r["suite"] == name], ("ttft_ms", "estimated_tpot_ms"))
            for name in suites
        }
    return {"metrics": metrics}


if __name__ == "__main__":
    accuracy.run("performance_test.sh",
                 lambda: accuracy.remote(measure_request, summarize, PREFIX),
                 lambda: accuracy.local(__doc__.splitlines()[0], select_inputs))
