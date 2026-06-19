# dejavu — prompt caching, proven live

> An AI repeats itself constantly. And every time, you pay for it.

## The intern and the 50-page contract

Imagine you hire an intern and hand them a 50-page contract. You ask them a question, and before answering, they read all 50 pages from page one. You ask another question — same thing, back to page one.

A large language model does exactly that. Before writing a single word it must "read" your entire prompt and build a mental map of who connects to what and where to look. This phase is called **prefill**, and it is the slow, expensive part — it happens before the response even starts.

**Prompt caching** is telling the intern: keep the contract open, with your notes already made. On the next question you do not re-read anything — you start from your notes.

The model saves that reading work. You send the same document with a new question, and the model does not re-read the 50 pages: it retrieves its cached notes and only processes the new question at the end. Less latency, less cost.

There is one rule to remember: **what does not change goes first; the new question goes last.** The model compares the prompt word by word from the beginning and stops at the first difference. If you put the question first, you throw away the cache on every turn.

On a large document you reuse many times, the difference is between paying once and paying every single time.

## What dejavu does

**dejavu** runs the same 10-question conversation **twice, concurrently**, against the Anthropic API and renders both sessions side-by-side in a Rich live terminal UI.

```
┌──────────────────────────────┬──────────────────────────────┐
│  Uncached panel              │  Cached panel                │
│  Running cost: $0.0412       │  Running cost: $0.0038       │
│  Turn 4  Cache read: 18 432  │  Turn 4  Cache read: 18 432  │
│  user: What is clause 12?    │  user: What is clause 12?    │
│  assistant: ...              │  assistant: ...              │
└──────────────────────────────┴──────────────────────────────┘
         10.8x cheaper   Saved: $0.0374
```

- **Left panel (uncached):** no `cache_control` marker — every turn re-pays the full base-input price for the entire contract plus the growing transcript. Cost climbs each round.
- **Right panel (cached, rolling breakpoint):** a `cache_control` ephemeral marker sits on the last message of the accumulated history each turn, so the entire prior prefix is cached. Only the newest question is fresh. After turn 1 the cost collapses to roughly 1/10 of the uncached price.

**Teaching beat:** on turn 1 the cached panel is slightly *more* expensive (the cache-write premium is +25%). From turn 2 onward the lines cross and the gap fans out in real time — the visual story of caching paying off.

## Install

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# Edit .env and set:
# ANTHROPIC_API_KEY=sk-ant-...
```

The key is read exclusively from the environment / `.env` file. It is never committed to source.

## Usage

```bash
dejavu [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `opus-4.8` | Model: `opus-4.8` / `fable-5` / `mythos-5` |
| `--doc` | *(built-in ~50-page contract)* | Path to a custom contract document |
| `--questions-file` | *(built-in 10 questions)* | Path to a custom questions JSON file |
| `--ttl` | `5m` | Cache TTL: `5m` (5-minute) or `1h` (1-hour) |
| `--max-tokens` | `400` | Max output tokens per turn |
| `--delay` | `0.0` | Pacing delay between turns (seconds, useful for recording) |

Before firing any real API calls, dejavu prints a **worst-case cost estimate** and asks for confirmation.

## Sample run

```bash
# Default: Opus 4.8, built-in contract, 10 questions, 5-minute cache TTL
dejavu

# Custom model and cache TTL
dejavu --model opus-4.8 --ttl 5m --max-tokens 400 --delay 1.0

# Custom contract document and questions
dejavu --doc ./my_contract.txt --questions-file ./my_questions.json
```

The terminal shows the two panels updating in real time. At the footer a cumulative **Nx cheaper** multiplier and **total $ saved** figure update each turn. On a phone-recorded terminal the numbers are large, high-contrast, and column-aligned.

## Metrics attribution

Two different sources provide the token counts that feed the cost engine:

| Metric | Source |
|---|---|
| `cached_input_tokens` (cache reads) | BAML `Collector` — `collector.usage.cached_input_tokens` |
| `cache_creation_input_tokens` (cache writes) | Raw Anthropic HTTP response body — `log.calls[-1].http_response.body.json()["usage"]["cache_creation_input_tokens"]` |

The BAML `Collector` exposes cache-read tokens directly in its typed usage object. Cache-write tokens (`cache_creation_input_tokens`) are **not** part of the typed usage — they must be extracted from the raw HTTP response. This is documented in the [BAML Collector docs](https://docs.boundaryml.com/guide/baml-advanced/collector-track-tokens).

## Pricing

Prices are fixed in `dejavu/pricing.py` and come from Anthropic's published pricing page (per MTok = per 1 000 000 tokens):

| Model | Base input | 5m cache write | 1h cache write | Cache read | Output |
|---|---|---|---|---|---|
| **Claude Opus 4.8** (default) | $5.00 | $6.25 | $10.00 | $0.50 | $25.00 |
| Claude Fable 5 | $10.00 | $12.50 | $20.00 | $1.00 | $50.00 |
| Claude Mythos 5 *(limited availability)* | $10.00 | $12.50 | $20.00 | $1.00 | $50.00 |

Three models are supported; two cache tiers (`5m` and `1h`). Mythos 5 is listed for completeness but is limited-availability — dejavu never defaults to it.

## Security

- The API key lives only in `ANTHROPIC_API_KEY` (env var) or `.env` (gitignored).
- `.env.example` ships without any real key.
- No secrets ever appear in source.
