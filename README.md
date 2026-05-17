# TRIP — Token Round-trip Inspection Profiler

Async LLM API benchmarking tool. Streams raw SSE via `httpx`, counts tokens with `tiktoken`, and generates Markdown + HTML reports with TTFT, TPS, and latency metrics across any OpenAI-compatible endpoint.

Configuration lives in `models.json` and `.env` — no code edits needed.

## Setup

```bash
pip install httpx tiktoken
cp .env.example .env   # then fill in your API keys
```

## Usage

```bash
# Default models (from run_by_default in models.json)
python trip.py

# ALL configured models
python trip.py --all

# All models from one or more providers
python trip.py --provider opencode
python trip.py --provider opencode xai

# Specific models by key
python trip.py --models opencode-deepseek-v4-flash xai-grok-4.3

# Concurrency (N simultaneous requests per model)
python trip.py --provider opencode -c 5

# Test a single model under load with 10 concurrent requests
python trip.py --models opencode-deepseek-v4-flash --concurrency 10

# Compare providers at 5x concurrency
python trip.py --provider opencode xai -c 5

# Custom prompt and token limit
python trip.py --all --max-tokens 128 --prompt "Write a haiku about GPUs"

# Debug mode
python trip.py --models opencode-deepseek-v4-flash --verbose

# Custom HTML report path
python trip.py --all --html ./results.html
```

### Selection priority

| Flag | Behavior |
|---|---|
| `--all` | Every model in `models.json` |
| `--provider opencode` | All models from that provider |
| `--provider opencode xai` | All models from both providers |
| `--models key1 key2` | Only the specified keys |
| *(none)* | Models listed in `run_by_default` |

### All flags

| Flag | Description | Default |
|---|---|---|
| `--all` | Benchmark every model in `models.json` | — |
| `--provider ...` | Benchmark all models from one or more providers | — |
| `--models key1 key2` | Benchmark specific model keys | `run_by_default` from config |
| `-c` / `--concurrency N` | Simultaneous requests per model | `1` |
| `--max-tokens N` | Max output tokens per request | `512` |
| `--temperature N` | Sampling temperature | `0.7` |
| `--prompt "..."` | Custom user prompt | Built-in prompt |
| `--system-prompt "..."` | Optional system prompt | — |
| `--timeout N` | Per-request timeout in seconds | `120` |
| `--html PATH` | Custom HTML report path | `trip_report_YYYYMMDD_HHMMSS.html` |
| `-v` / `--verbose` | Debug: show request URLs, auth, response bodies | — |

## Configuration

### `models.json`

Defines providers and models. To add a new provider:

```json
"together": {
  "base_url": "https://api.together.xyz/v1",
  "api_key_env": "TOGETHER_API_KEY",
  "endpoint_type": "chat"
}
```

Then add models pointing to it:

```json
{ "key": "together-llama3-70b", "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "provider": "together" }
```

Add the key to `.env` and it just works — zero code changes.

### `.env`

API keys referenced by `api_key_env` in `models.json`:

```
OPENCODE_API_KEY=sk-...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-...
LOCAL_API_KEY=
```

Keys are auto-loaded from `.env`. Environment variables take precedence.

### LM Studio (local models)

1. Open LM Studio, load a model, start the server (default port 1234)
2. The `local` provider in `models.json` is pre-configured for `http://localhost:1234/v1`
3. Run:

```bash
python trip.py --provider local
```

## Output

- **Terminal**: Markdown table with per-model TTFT, TPS, tokens, and errors
- **HTML**: Auto-generated `trip_report_YYYYMMDD_HHMMSS.html` — timestamped so previous runs are preserved

## Providers & models

| Provider | Models | Key prefix |
|---|---|---|
| OpenCode | 15 (deepseek, minimax, kimi, glm, qwen, mimo, hy3) | `opencode-` |
| xAI | 3 (grok-4.20, grok-4.3) | `xai-` |
| OpenRouter | 10 (claude, gpt-5, deepseek, gemini, llama, qwen) | `or-` |
| Local | 1 (llama via LM Studio) | `local-` |

Run `python trip.py --all` to benchmark everything, or `--provider openrouter` for a specific provider.