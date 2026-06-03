#!/usr/bin/env python3
"""
TRIP — Token Round-trip Inspection Profiler

Async LLM API benchmarking tool.  Streams raw SSE via httpx, counts tokens
with tiktoken, and generates Markdown + HTML reports with TTFT, TPS, and
latency metrics across any OpenAI-compatible endpoint.

Configuration lives in models.json and .env — no code edits needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import httpx

_DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_dotenv(path: str = _DOTENV_PATH) -> None:
    """Load a .env file into os.environ.  Existing env vars take precedence."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# ---------------------------------------------------------------------------
# Configuration — loaded from models.json + .env
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")


def _load_config(path: str = _CONFIG_PATH) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Load providers + models from ``models.json`` and resolve into a flat
    MODEL_MATRIX dict.  Returns (matrix, run_by_default).
    """
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    providers: dict[str, dict] = cfg.get("providers", {})
    run_by_default: list[str] = cfg.get("run_by_default", [])

    matrix: dict[str, dict[str, str]] = {}
    for entry in cfg.get("models", []):
        key = entry["key"]
        provider_name = entry.get("provider", "")
        provider = providers.get(provider_name, {})
        matrix[key] = {
            "model_id": entry.get("model_id", key),
            "base_url": entry.get("base_url", provider.get("base_url", "")),
            "api_key_env": entry.get("api_key_env", provider.get("api_key_env", "")),
            "endpoint_type": entry.get("endpoint_type", provider.get("endpoint_type", "chat")),
            "_provider": provider_name,
        }
        auth_type = entry.get("auth_type", provider.get("auth_type"))
        if auth_type:
            matrix[key]["auth_type"] = auth_type

    return matrix, run_by_default


MODEL_MATRIX, _DEFAULT_MODELS = _load_config()
MODELS_TO_RUN: list[str] = list(_DEFAULT_MODELS)


_ENDPOINT_PATHS: dict[str, str] = {
    "chat": "/chat/completions",
    "messages": "/messages",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    success: bool
    status_code: int | None
    ttft: float
    tokens_per_second: float
    total_latency: float
    output_tokens: int
    input_tokens: int
    total_tokens: int
    error: str | None
    request_id: int


@dataclass
class ModelBenchResult:
    """Aggregated result for a single model across C concurrent runs."""
    model_key: str
    target_id: str
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    tps_mean: float
    tps_p50: float
    tps_p95: float
    output_tokens_mean: float
    total_tokens_mean: float
    total_latency_mean: float
    requests_sent: int
    successes: int
    errors: list[str]
    endpoint_type: str
    provider: str


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, Callable[[str], int]] = {}


def _build_tiktoken_counter(model: str) -> Callable[[str], int]:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text, disallowed_special=()))


def _build_fallback_counter() -> Callable[[str], int]:
    return lambda text: max(1, len(text) // 4)


def _get_tokenizer(model: str) -> Callable[[str], int]:
    if model not in _TOKENIZER_CACHE:
        if HAS_TIKTOKEN:
            _TOKENIZER_CACHE[model] = _build_tiktoken_counter(model)
        else:
            _TOKENIZER_CACHE[model] = _build_fallback_counter()
    return _TOKENIZER_CACHE[model]


def count_tokens(text: str, model: str) -> int:
    return _get_tokenizer(model)(text)


# ---------------------------------------------------------------------------
# SSE parsing  (format-agnostic)
# ---------------------------------------------------------------------------

async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[dict]:
    """Yield parsed JSON dicts from a raw SSE stream.

    Handles both bare ``data: {...}`` lines and the Anthropic-style
    ``event: ...`` / ``data: {...}`` pair format.  Stops at ``[DONE]``.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        # Consume (but don't use) optional event-type lines
        if line.startswith("event: "):
            continue
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            return
        if not payload:
            continue
        yield json.loads(payload)


# ---------------------------------------------------------------------------
# Content extractors  (one per endpoint_type)
# ---------------------------------------------------------------------------

def _extract_chat_content(event: dict) -> str | None:
    """OpenAI ``/chat/completions`` content (including DeepSeek reasoning_content)."""
    choices = event.get("choices")
    if not choices:
        return None
    delta = choices[0].get("delta", {})
    content = delta.get("content")
    if content:
        return content
    reasoning = delta.get("reasoning_content")
    if reasoning:
        return reasoning
    return None


def _extract_messages_content(event: dict) -> str | None:
    """Anthropic ``/messages`` content block delta."""
    if event.get("type") == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text")
    return None


_CONTENT_EXTRACTORS: dict[str, Callable[[dict], str | None]] = {
    "chat": _extract_chat_content,
    "messages": _extract_messages_content,
}


# ---------------------------------------------------------------------------
# Per-request benchmark
# ---------------------------------------------------------------------------

_AUTH_HEADERS: dict[str, str] = {
    "bearer": "Authorization",
    "x-api-key": "x-api-key",
}

_VERBOSE = False


async def benchmark_request(
    *,
    request_id: int,
    client: httpx.AsyncClient,
    url: str,
    api_key: str | None,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
    content_extractor: Callable[[dict], str | None],
    auth_type: str = "bearer",
) -> RequestResult:

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        header_name = _AUTH_HEADERS.get(auth_type, "Authorization")
        header_value = api_key if auth_type == "x-api-key" else f"Bearer {api_key}"
        headers[header_name] = header_value

    if _VERBOSE:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if api_key and len(api_key) > 10 else "(none)"
        print(f"    [DEBUG] POST {url}")
        print(f"    [DEBUG] model={model}  auth={auth_type}  key={masked_key}")
        print(f"    [DEBUG] headers_sent={list(headers.keys())}")

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    # Temperature is not supported by every backend; skip if zero
    if temperature > 0:
        body["temperature"] = temperature

    accumulated = ""
    first_chunk_time: float | None = None
    start_time = time.perf_counter()
    input_tok: int = 0
    _api_completion_tokens: int = 0
    error: str | None = None
    status_code: int | None = None
    ttft = 0.0
    generation_time = 0.0
    total_latency = 0.0

    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            status_code = resp.status_code
            if status_code != 200:
                error = f"HTTP {status_code}"
                body_text = await resp.aread()
                if _VERBOSE:
                    print(f"    [DEBUG] response_body={body_text.decode(errors='replace')[:500]}")
                try:
                    detail = json.loads(body_text)
                    msg = detail.get("error", {}).get("message", "")
                    if not msg:
                        msg = detail.get("error", "")
                        if isinstance(msg, dict):
                            msg = msg.get("message", str(msg))
                    error += f": {msg}" if msg else ""
                except Exception:
                    error += f": {body_text.decode(errors='replace')[:200]}"
                return RequestResult(
                    success=False, status_code=status_code,
                    ttft=0.0, tokens_per_second=0.0,
                    total_latency=time.perf_counter() - start_time,
                    output_tokens=0, input_tokens=0, total_tokens=0,
                    error=error, request_id=request_id,
                )

            async for event in _iter_sse_events(resp):
                now = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = now
                    ttft = first_chunk_time - start_time

                content = content_extractor(event)
                if content:
                    accumulated += content

                # Some providers include usage in the final delta event
                usage = event.get("usage")
                if usage:
                    input_tok = input_tok or usage.get("prompt_tokens", 0)
                    api_output_tok = usage.get("completion_tokens")
                    if api_output_tok:
                        _api_completion_tokens = api_output_tok

            end_time = time.perf_counter()
            total_latency = end_time - start_time
            generation_time = (
                end_time - first_chunk_time
                if first_chunk_time is not None
                else total_latency
            )

            # If streaming returned no content, fall back to a non-streaming request
            # to get the actual output and measure end-to-end latency.
            if not accumulated and first_chunk_time is None:
                if _VERBOSE:
                    print(f"    [DEBUG] Stream produced no content — retrying non-streaming")
                nf_body = dict(body)
                nf_body["stream"] = False
                try:
                    nf_resp = await client.post(url, json=nf_body, headers=headers)
                    nf_total = time.perf_counter() - start_time
                    if nf_resp.status_code == 200:
                        nf_data = nf_resp.json()
                        nf_content = ""
                        nf_choices = nf_data.get("choices", [])
                        if nf_choices:
                            nf_content = nf_choices[0].get("message", {}).get("content", "") or ""
                            # DeepSeek reasoning_content in non-streaming
                            nf_reasoning = nf_choices[0].get("message", {}).get("reasoning_content", "") or ""
                            nf_content = nf_content + nf_reasoning
                        accumulated = nf_content
                        total_latency = nf_total
                        generation_time = nf_total
                        ttft = 0.0
                        nf_usage = nf_data.get("usage", {})
                        input_tok = nf_usage.get("prompt_tokens", input_tok)
                        if nf_usage.get("completion_tokens"):
                            _api_completion_tokens = nf_usage["completion_tokens"]
                except Exception:
                    pass

            output_tok = count_tokens(accumulated, model) if accumulated else 0
            # Prefer the API's own token count if we got one
            if _api_completion_tokens and _api_completion_tokens > output_tok:
                output_tok = _api_completion_tokens
            if input_tok == 0:
                prompt_text = json.dumps(messages, ensure_ascii=False)
                input_tok = count_tokens(prompt_text, model)

            tps = output_tok / generation_time if generation_time > 0 else 0.0

            return RequestResult(
                success=True, status_code=200,
                ttft=ttft, tokens_per_second=tps,
                total_latency=total_latency,
                output_tokens=output_tok,
                input_tokens=input_tok,
                total_tokens=input_tok + output_tok,
                error=None, request_id=request_id,
            )

    except httpx.TimeoutException:
        error = "timeout"
    except httpx.HTTPStatusError as exc:
        error = f"HTTP {exc.response.status_code}"
        status_code = exc.response.status_code
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    total_latency = time.perf_counter() - start_time
    return RequestResult(
        success=False, status_code=status_code,
        ttft=0.0, tokens_per_second=0.0,
        total_latency=total_latency,
        output_tokens=0, input_tokens=0, total_tokens=0,
        error=error, request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    k = (p / 100.0) * (len(sorted_data) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


@dataclass
class _Stats:
    mean: float
    p50: float
    p95: float
    count: int


def _compute_stats(values: list[float]) -> _Stats:
    if not values:
        return _Stats(0.0, 0.0, 0.0, 0)
    n = len(values)
    s = sorted(values)
    return _Stats(
        mean=sum(s) / n,
        p50=_percentile(s, 50),
        p95=_percentile(s, 95),
        count=n,
    )


# ---------------------------------------------------------------------------
# Model-level benchmark runner
# ---------------------------------------------------------------------------

async def benchmark_model(
    *,
    model_key: str,
    config: dict[str, str],
    prompts: list[str],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    runs: int,
    timeout: float,
    provider_name: str = "",
) -> ModelBenchResult:
    """Run *runs* rounds of *concurrency* concurrent requests against a single
    model, cycling through *prompts*.  A warm-up request is sent first to
    eliminate DNS / TLS / connection overhead."""

    endpoint_type = config["endpoint_type"]
    base_url = config["base_url"].rstrip("/")
    url = base_url + _ENDPOINT_PATHS[endpoint_type]
    api_key = os.getenv(config["api_key_env"])
    model_id = config["model_id"]

    if _VERBOSE:
        masked = f"{api_key[:6]}...{api_key[-4:]}" if api_key and len(api_key) > 10 else "(empty)"
        print(f"    [DEBUG] model_key={model_key}  url={url}  model_id={model_id}")
        print(f"    [DEBUG] api_key_env=${config['api_key_env']}  key={masked}  auth_type={config.get('auth_type', 'bearer')}")

    if not api_key:
        print(f"    WARNING: ${config['api_key_env']} not set — request will be unauthenticated")

    extractor = _CONTENT_EXTRACTORS.get(endpoint_type)
    if extractor is None:
        return ModelBenchResult(
            model_key=model_key, target_id=model_id,
            ttft_mean=0, ttft_p50=0, ttft_p95=0,
            tps_mean=0, tps_p50=0, tps_p95=0,
            output_tokens_mean=0, total_tokens_mean=0,
            total_latency_mean=0,
            requests_sent=concurrency * runs, successes=0,
            errors=[f"Unknown endpoint_type: {endpoint_type}"],
            endpoint_type=endpoint_type,
            provider=provider_name,
        )

    limits = httpx.Limits(
        max_keepalive_connections=concurrency,
        max_connections=concurrency,
    )

    auth_type = config.get("auth_type", "bearer")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        header_name = _AUTH_HEADERS.get(auth_type, "Authorization")
        header_value = api_key if auth_type == "x-api-key" else f"Bearer {api_key}"
        headers[header_name] = header_value

    async with httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(timeout),
    ) as client:
        # ---- warm-up ----
        try:
            wu_body = {
                "model": model_id,
                "messages": [{"role": "user", "content": "warm-up"}],
                "max_tokens": 16,
                "stream": True,
            }
            if temperature > 0:
                wu_body["temperature"] = temperature
            async with client.stream("POST", url, json=wu_body, headers=headers) as wu_resp:
                if wu_resp.status_code == 200:
                    async for _ in _iter_sse_events(wu_resp):
                        pass
        except Exception:
            pass

        # ---- benchmark runs ----
        all_results: list[RequestResult] = []
        for run_idx in range(runs):
            prompt = prompts[run_idx % len(prompts)]
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            tasks = [
                benchmark_request(
                    request_id=run_idx * concurrency + i + 1,
                    client=client,
                    url=url,
                    api_key=api_key,
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    content_extractor=extractor,
                    auth_type=auth_type,
                )
                for i in range(concurrency)
            ]
            run_results = await asyncio.gather(*tasks)
            all_results.extend(run_results)

    success_results = [r for r in all_results if r.success]
    errors = [r.error for r in all_results if r.error]

    if success_results:
        ttft_vals = [r.ttft for r in success_results]
        tps_vals = [r.tokens_per_second for r in success_results]
        out_tok_vals = [r.output_tokens for r in success_results]
        tot_tok_vals = [r.total_tokens for r in success_results]
        lat_vals = [r.total_latency for r in success_results]

        ttft_s = _compute_stats(ttft_vals)
        tps_s = _compute_stats(tps_vals)
        output_tok_mean = sum(out_tok_vals) / len(out_tok_vals)
        total_tok_mean = sum(tot_tok_vals) / len(tot_tok_vals)
        lat_mean = sum(lat_vals) / len(lat_vals)
    else:
        ttft_s = _Stats(0, 0, 0, 0)
        tps_s = _Stats(0, 0, 0, 0)
        output_tok_mean = 0.0
        total_tok_mean = 0.0
        lat_mean = 0.0

    return ModelBenchResult(
        model_key=model_key,
        target_id=model_id,
        ttft_mean=ttft_s.mean,
        ttft_p50=ttft_s.p50,
        ttft_p95=ttft_s.p95,
        tps_mean=tps_s.mean,
        tps_p50=tps_s.p50,
        tps_p95=tps_s.p95,
        output_tokens_mean=output_tok_mean,
        total_tokens_mean=total_tok_mean,
        total_latency_mean=lat_mean,
        requests_sent=concurrency * runs,
        successes=len(success_results),
        errors=errors,
        endpoint_type=endpoint_type,
        provider=provider_name,
    )


# ---------------------------------------------------------------------------
# Output — Markdown table
# ---------------------------------------------------------------------------

def _fmt(s: float, decimals: int = 3) -> str:
    return f"{s:.{decimals}f}"


def print_model_table(results: list[ModelBenchResult]) -> None:
    """Print a Markdown summary table of per-model results to stdout."""

    header = (
        "| Model | Provider | Target ID | TTFT mean (s) | TTFT p50 (s) | "
        "TTFT p95 (s) | TPS mean | TPS p50 | TPS p95 | Out Tok | "
        "Lat avg (s) | OK | Errors |"
    )
    sep = (
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | :--- |"
    )

    print()
    print(header)
    print(sep)

    for r in results:
        err_str = ", ".join(r.errors[:2])
        if len(r.errors) > 2:
            err_str += f" (+{len(r.errors) - 2} more)"
        ok_str = f"{r.successes}/{r.requests_sent}"

        print(
            f"| {r.model_key} | {r.provider} | {r.target_id} "
            f"| {_fmt(r.ttft_mean)} | {_fmt(r.ttft_p50)} | {_fmt(r.ttft_p95)} "
            f"| {_fmt(r.tps_mean, 1)} | {_fmt(r.tps_p50, 1)} | {_fmt(r.tps_p95, 1)} "
            f"| {_fmt(r.output_tokens_mean, 1)} "
            f"| {_fmt(r.total_latency_mean)} "
            f"| {ok_str}"
            f"| {err_str} |"
        )

    print()

    success_count = sum(1 for r in results if r.successes > 0)
    total_count = len(results)
    print(f"Models benchmarked: {success_count}/{total_count} succeeded\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _tps_color(tps: float) -> str:
    if tps >= 150:
        return "#22c55e"
    if tps >= 80:
        return "#84cc16"
    if tps >= 40:
        return "#eab308"
    if tps >= 20:
        return "#f97316"
    return "#ef4444"


def _ttft_color(ttft: float) -> str:
    if ttft <= 0.3:
        return "#22c55e"
    if ttft <= 0.6:
        return "#84cc16"
    if ttft <= 1.0:
        return "#eab308"
    if ttft <= 2.0:
        return "#f97316"
    return "#ef4444"


def _lat_color(lat: float) -> str:
    if lat <= 1.0:
        return "#22c55e"
    if lat <= 3.0:
        return "#84cc16"
    if lat <= 6.0:
        return "#eab308"
    if lat <= 10.0:
        return "#f97316"
    return "#ef4444"


def _bar_pct(value: float, max_val: float) -> float:
    return min(value / max_val, 1.0) * 100 if max_val > 0 else 0


def _timestamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_html_report(
    results: list[ModelBenchResult],
    *,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    concurrency: int,
    runs: int,
    output_path: str,
) -> None:
    """Write a self-contained HTML benchmark report to *output_path*."""

    max_tps = max((r.tps_mean for r in results if r.successes > 0), default=1) or 1
    max_lat = max((r.total_latency_mean for r in results if r.successes > 0), default=1) or 1
    ts = _timestamp()

    success_count = sum(1 for r in results if r.successes > 0)
    fail_count = len(results) - success_count

    rows_html = ""
    for r in results:
        is_ok = r.successes > 0
        ok_badge = (
            '<span class="badge ok">OK</span>'
            if is_ok else
            '<span class="badge fail">FAIL</span>'
        )
        tps_s = f"{r.tps_mean:.1f}" if is_ok else "&mdash;"
        ttft_s = f"{r.ttft_mean * 1000:.0f}&thinsp;ms" if is_ok else "&mdash;"
        lat_s = f"{r.total_latency_mean:.2f}&thinsp;s" if is_ok else "&mdash;"
        out_s = f"{r.output_tokens_mean:.0f}" if is_ok else "&mdash;"

        tps_bar = (
            f'<div class="bar" style="width:{_bar_pct(r.tps_mean, max_tps):.1f}%;'
            f'background:{_tps_color(r.tps_mean)}"></div>'
            if is_ok else ""
        )
        ttft_bar = (
            f'<div class="bar" style="width:{_bar_pct(r.ttft_mean, 2.0):.1f}%;'
            f'background:{_ttft_color(r.ttft_mean)}"></div>'
            if is_ok else ""
        )
        lat_bar = (
            f'<div class="bar" style="width:{_bar_pct(r.total_latency_mean, max_lat):.1f}%;'
            f'background:{_lat_color(r.total_latency_mean)}"></div>'
            if is_ok else ""
        )

        err_cell = ""
        if r.errors:
            shown = ", ".join(r.errors[:2])
            if len(r.errors) > 2:
                shown += f" (+{len(r.errors) - 2} more)"
            err_cell = f'<div class="error">{shown}</div>'

        rows_html += f"""      <tr class="{'row-ok' if is_ok else 'row-fail'}">
        <td>{r.model_key}<br><span class="sub">{r.target_id}</span></td>
        <td><span class="provider">{r.provider}</span></td>
        <td>{ok_badge} {r.successes}/{r.requests_sent}</td>
        <td>
          <div class="cell-bar">{ttft_bar}</div>
          {ttft_s}
          <span class="sub">p50 {r.ttft_p50 * 1000:.0f}&thinsp;ms</span>
          <span class="sub">p95 {r.ttft_p95 * 1000:.0f}&thinsp;ms</span>
        </td>
        <td>
          <div class="cell-bar">{tps_bar}</div>
          {tps_s}
          <span class="sub">p50 {r.tps_p50:.1f}</span>
          <span class="sub">p95 {r.tps_p95:.1f}</span>
        </td>
        <td>{out_s}</td>
        <td>
          <div class="cell-bar">{lat_bar}</div>
          {lat_s}
        </td>
        <td>{err_cell}</td>
      </tr>
"""

    prompt_previews = " | ".join(
        p[:80] + ("..." if len(p) > 80 else "") for p in prompts
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRIP &mdash; Token Round-trip Inspection Profiler &mdash; {ts}</title>
<style>
  :root {{
    --bg: #0b0f19;
    --surface: #131927;
    --border: #1e293b;
    --text: #e2e8f0;
    --dim: #64748b;
    --accent: #38bdf8;
    --ok: #22c55e;
    --fail: #ef4444;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); padding: 2rem; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: .25rem; }}
  .logo {{ color: var(--accent); }}
  .timestamp {{ color: var(--dim); font-size: .85rem; margin-bottom: 1.5rem; }}
  .meta {{ display: flex; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .meta-item {{ background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; padding: .6rem 1rem; }}
  .meta-label {{ font-size: .75rem; color: var(--dim); text-transform: uppercase; letter-spacing: .05em; }}
  .meta-value {{ font-size: 1.1rem; font-weight: 600; margin-top: .15rem; }}
  .prompt {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: .75rem 1rem; margin-bottom: 1.5rem;
             font-size: .85rem; color: var(--dim); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  thead th {{ text-align: left; padding: .6rem .75rem; border-bottom: 2px solid var(--border);
              color: var(--dim); font-weight: 600; text-transform: uppercase;
              font-size: .7rem; letter-spacing: .08em; white-space: nowrap; }}
  tbody td {{ padding: .65rem .75rem; border-bottom: 1px solid var(--border);
              vertical-align: top; }}
  .row-ok td {{ background: var(--surface); }}
  .row-fail td {{ background: #1a1015; }}
  .sub {{ display: block; font-size: .7rem; color: var(--dim); }}
  .badge {{ display: inline-block; font-size: .65rem; font-weight: 700;
            padding: .15em .45em; border-radius: 4px; text-transform: uppercase;
            letter-spacing: .06em; margin-right: .35rem; }}
  .badge.ok {{ background: rgba(34,197,94,.15); color: var(--ok); }}
  .badge.fail {{ background: rgba(239,68,68,.15); color: var(--fail); }}
  .cell-bar {{ height: 4px; background: var(--border); border-radius: 2px;
               margin-bottom: .3rem; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: 2px; transition: width .3s; }}
  .error {{ color: var(--fail); font-size: .75rem; margin-top: .25rem; }}
  .provider {{ background: rgba(56,189,248,.12); color: var(--accent);
               font-size: .7rem; font-weight: 600; padding: .15em .45em;
               border-radius: 4px; text-transform: uppercase; letter-spacing: .04em; }}
  .summary {{ margin-top: 1.5rem; display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .summary-card {{ background: var(--surface); border: 1px solid var(--border);
                    border-radius: 8px; padding: 1rem 1.5rem; flex: 1; min-width: 200px; }}
  .summary-card .val {{ font-size: 1.6rem; font-weight: 700; }}
  .summary-card .label {{ font-size: .75rem; color: var(--dim); text-transform: uppercase;
                           letter-spacing: .06em; margin-top: .25rem; }}
</style>
</head>
<body>
<div class="container">
  <h1><span class="logo">TRIP</span> — Benchmarks</h1>
  <p class="timestamp">{ts}</p>

  <div class="meta">
    <div class="meta-item"><div class="meta-label">Models</div>
      <div class="meta-value">{success_count}/{len(results)} succeeded</div></div>
    <div class="meta-item"><div class="meta-label">Concurrency</div>
      <div class="meta-value">{concurrency}</div></div>
    <div class="meta-item"><div class="meta-label">Runs</div>
      <div class="meta-value">{runs}</div></div>
    <div class="meta-item"><div class="meta-label">Max Tokens</div>
      <div class="meta-value">{max_tokens}</div></div>
    <div class="meta-item"><div class="meta-label">Temperature</div>
      <div class="meta-value">{temperature}</div></div>
  </div>

  <div class="prompt"><strong>Prompt(s):</strong> {prompt_previews}</div>

  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Provider</th>
        <th>Status</th>
        <th>TTFT</th>
        <th>Tokens/sec</th>
        <th>Out Tok</th>
        <th>Latency</th>
        <th>Errors</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>

  <div class="summary">
""" 

    if success_count > 0:
        ok = [r for r in results if r.successes > 0]
        best_tps = max(ok, key=lambda r: r.tps_mean)
        best_ttft = min(ok, key=lambda r: r.ttft_mean)
        best_lat = min(ok, key=lambda r: r.total_latency_mean)

        html += f"""    <div class="summary-card">
      <div class="val" style="color:{_tps_color(best_tps.tps_mean)}">{best_tps.tps_mean:.1f}</div>
      <div class="label">Best TPS &mdash; {best_tps.model_key}</div>
    </div>
    <div class="summary-card">
      <div class="val" style="color:{_ttft_color(best_ttft.ttft_mean)}">{best_ttft.ttft_mean * 1000:.0f}&thinsp;ms</div>
      <div class="label">Lowest TTFT &mdash; {best_ttft.model_key}</div>
    </div>
    <div class="summary-card">
      <div class="val" style="color:{_lat_color(best_lat.total_latency_mean)}">{best_lat.total_latency_mean:.2f}&thinsp;s</div>
      <div class="label">Lowest Latency &mdash; {best_lat.model_key}</div>
    </div>
"""

    html += f"""  </div>
</div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    print(f"  HTML report: file://{abs_path}")

    try:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_matrix_benchmark(
    *,
    model_keys: list[str],
    prompts: list[str],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    runs: int,
    timeout: float,
) -> list[ModelBenchResult]:
    """Iterate over *model_keys*, benchmark each, return results."""

    missing_keys: dict[str, str] = {}
    for key in model_keys:
        cfg = MODEL_MATRIX.get(key)
        if cfg:
            env_var = cfg["api_key_env"]
            if not os.getenv(env_var):
                missing_keys[key] = env_var
    if missing_keys:
        print("  WARNING — missing API keys:")
        for k, env_var in missing_keys.items():
            print(f"    {k}: set ${env_var} or add it to .env")
        print()

    results: list[ModelBenchResult] = []
    for key in model_keys:
        config = MODEL_MATRIX.get(key)
        if config is None:
            print(f"  [SKIP]  unknown model key: {key}")
            continue

        print(f"  {key} ({config['model_id']})  [{config['endpoint_type']}]  "
              f"@ {config.get('_provider', '?')}  runs={runs}  concurrency={concurrency}  "
              f"max_tokens={max_tokens}  ...", end=" ")
        sys.stdout.flush()

        result = await benchmark_model(
            model_key=key,
            config=config,
            prompts=prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            runs=runs,
            timeout=timeout,
            provider_name=config.get("_provider", ""),
        )

        status = (
            f"OK ({result.successes}/{result.requests_sent})"
            if result.successes > 0
            else "FAIL"
        )
        print(status)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = (
    "Explain the concept of 'attention is all you need' in the context of "
    "transformer architectures, including how self-attention, multi-head "
    "attention, and positional encodings work together. Provide a thorough "
    "yet concise technical explanation."
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TRIP — Token Round-trip Inspection Profiler. Benchmark LLM APIs for TTFT, TPS, and latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run default models (from models.json run_by_default)\n"
            "  python benchmark.py\n\n"
            "  # Run ALL configured models\n"
            "  python benchmark.py --all\n\n"
            "  # Run all models from a specific provider\n"
            "  python benchmark.py --provider opencode\n"
            "  python benchmark.py --provider opencode xai\n\n"
            "  # Run specific models with concurrency\n"
            "  python benchmark.py --models opencode-deepseek-v4-flash xai-grok-4.3 -c 5\n\n"
            "  # Combine provider filter with specific models\n"
            "  python benchmark.py --provider opencode --models xai-grok-4.3\n\n"
            "  # Quick smoke test\n"
            "  python benchmark.py --provider xai --max-tokens 64\n"
        ),
    )
    p.add_argument("--models", nargs="+", default=None,
                    help="Specific model keys to benchmark.")
    p.add_argument("--all", action="store_true",
                    help="Benchmark ALL models defined in models.json.")
    p.add_argument("--provider", nargs="+", default=None,
                    help="Benchmark all models from one or more providers (e.g. opencode xai).")
    p.add_argument("--prompt", default=_DEFAULT_PROMPT,
                    help="User prompt(s) to send. Multiple values cycle across runs.")
    p.add_argument("--runs", type=int, default=1,
                    help="Number of benchmark runs per model (default: 1).")
    p.add_argument("--system-prompt", default=None,
                    help="Optional system prompt.")
    p.add_argument("--max-tokens", type=int, default=512,
                    help="Maximum output tokens per request (default: 512).")
    p.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (default: 0.7).")
    p.add_argument("-c", "--concurrency", type=int, default=1,
                    help="Concurrent requests per model (default: 1).")
    p.add_argument("--timeout", type=float, default=120.0,
                    help="Per-request timeout in seconds (default: 120).")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Print debug info: request URLs, masked keys, response bodies.")
    p.add_argument("--html", nargs="?", const="__default__", default=None,
                    help="HTML report path (default: trip_report.html next to script). Always generated.")
    return p


def main() -> None:
    global _VERBOSE
    parser = _build_parser()
    args = parser.parse_args()

    _VERBOSE = args.verbose

    # Resolve model selection: --all > --provider > --models > defaults
    if args.all:
        model_keys = list(MODEL_MATRIX.keys())
    elif args.provider:
        providers = set(args.provider)
        model_keys = [
            k for k, v in MODEL_MATRIX.items()
            if v.get("_provider", "") in providers
        ]
        unknown_providers = providers - {v.get("_provider", "") for v in MODEL_MATRIX.values()}
        if unknown_providers:
            available = sorted({v.get("_provider", "") for v in MODEL_MATRIX.values()})
            parser.error(f"Unknown provider(s): {', '.join(unknown_providers)}. Available: {', '.join(available)}")
    elif args.models:
        model_keys = args.models
    else:
        model_keys = list(MODELS_TO_RUN)

    unresolved = [k for k in model_keys if k not in MODEL_MATRIX]
    if unresolved:
        parser.error(f"Unknown model key(s): {', '.join(unresolved)}")

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")

    prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]

    print(f"TRIP — Token Round-trip Inspection Profiler")
    print(f"  Models:      {', '.join(model_keys)}")
    print(f"  Concurrency: {args.concurrency}  |  Max tokens: {args.max_tokens}  "
          f"|  Temp: {args.temperature}  |  Runs: {args.runs}")
    print()

    results = asyncio.run(
        run_matrix_benchmark(
            model_keys=model_keys,
            prompts=prompts,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            runs=args.runs,
            timeout=args.timeout,
        )
    )

    print_model_table(results)

    # HTML report — always generated; --html overrides the path
    html_path = args.html if args.html not in (None, "__default__") else None
    if html_path is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trip_report_{ts}.html")
    generate_html_report(
        results,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        runs=args.runs,
        output_path=html_path,
    )

    sys.exit(0 if any(r.successes > 0 for r in results) else 1)


if __name__ == "__main__":
    main()
