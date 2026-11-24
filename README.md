# llm-inference-bench

A deliberately small harness for measuring what a served LLM endpoint actually does, using only the Python standard library.

It speaks the OpenAI-compatible /v1/chat/completions shape over SSE streaming, so the same script runs unchanged against vLLM on your own box, a rented GPU host, or a managed platform like Baseten. That is the point: the numbers should be comparable, and the client should not be the variable.

## What it measures

- TTFT - time to first token, what a user feels before any output appears
- Inter-token latency (ITL) - gap between chunks, i.e. how smoothly text streams
- Output throughput - tokens per second per request
- Percentiles p50 / p90 / p99 instead of a mean, because inference latency is not normally distributed

## Usage

    python bench.py --base-url http://localhost:8000 --model your-model \
        --requests 32 --concurrency 4 --max-tokens 128

Against a hosted endpoint:

    python bench.py --base-url https://your-endpoint.example.com --model your-model \
        --api-key $BASETEN_API_KEY --requests 64 --concurrency 8

Results print as a per-request table plus an aggregate summary, and can be written to JSON with --json out.json.

## Caveats

- Cold starts are tagged, not averaged away - the first request is reported separately so warm-up cost stays visible.
- --warmup runs throwaway requests before the timed phase.
- TTFT measured from a client includes network RTT. Irrelevant same-host, not irrelevant across regions: keep the client close or subtract a baseline ping yourself.
- Servers chunk differently. ITL is per chunk, so a server batching 4 tokens per chunk looks smoother than one flushing per token at equal throughput.

## Status

Personal tooling, evolving as I hit cases it does not cover.
