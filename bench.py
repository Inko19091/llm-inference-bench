#!/usr/bin/env python3
"""Measure TTFT, inter-token latency and throughput against an OpenAI-compatible endpoint."""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def one_request(base_url, model, prompt, max_tokens, api_key, timeout):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }).encode()

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    req = urllib.request.Request(url, data=body, headers=headers)
    start = time.perf_counter()
    first = None
    gaps = []
    tokens = 0
    last = start

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            chunk = raw.decode("utf-8", "replace").strip()
            if not chunk.startswith("data:"):
                continue
            payload = chunk[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if not delta.get("content"):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            else:
                gaps.append(now - last)
            last = now
            tokens += 1

    end = time.perf_counter()
    if first is None:
        return {"ok": False, "error": "no tokens received"}

    return {
        "ok": True,
        "ttft": first - start,
        "total": end - start,
        "tokens": tokens,
        "tps": tokens / (end - first) if end > first else 0.0,
        "itl": statistics.mean(gaps) * 1000 if gaps else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain in three sentences why p99 latency matters more than the mean.")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    for _ in range(args.warmup):
        try:
            one_request(args.base_url, args.model, args.prompt, 16, args.api_key, args.timeout)
        except Exception as exc:
            print("warmup failed:", exc)

    results = []
    lock = threading.Lock()
    sem = threading.Semaphore(args.concurrency)

    def worker(index):
        with sem:
            try:
                res = one_request(args.base_url, args.model, args.prompt, args.max_tokens, args.api_key, args.timeout)
            except urllib.error.HTTPError as exc:
                res = {"ok": False, "error": "HTTP " + str(exc.code)}
            except Exception as exc:
                res = {"ok": False, "error": str(exc)}
            res["index"] = index
            with lock:
                results.append(res)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.requests)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started

    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)

    for r in sorted(results, key=lambda x: x["index"]):
        if r.get("ok"):
            print("{:>3}  ttft {:7.3f}s  itl {:7.1f}ms  {:6.1f} tok/s  n={}".format(r["index"], r["ttft"], r["itl"], r["tps"], r["tokens"]))
        else:
            print("{:>3}  FAIL  {}".format(r["index"], r["error"]))

    ttfts = [r["ttft"] for r in ok]
    itls = [r["itl"] for r in ok]
    tpss = [r["tps"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)

    summary = {
        "requests": len(results),
        "failed": failed,
        "wall_seconds": wall,
        "aggregate_tokens_per_second": total_tokens / wall if wall else 0.0,
        "ttft_ms": {
            "p50": percentile(ttfts, 50) * 1000,
            "p90": percentile(ttfts, 90) * 1000,
            "p99": percentile(ttfts, 99) * 1000,
        },
        "itl_ms": {
            "p50": percentile(itls, 50),
            "p90": percentile(itls, 90),
            "p99": percentile(itls, 99),
        },
    }

    print()
    print("ok {}  failed {}".format(len(ok), failed))
    print("aggregate: {:.1f} tok/s over {:.2f}s wall".format(summary["aggregate_tokens_per_second"], wall))
    print("ttft p50/p90/p99 ms: {:.0f} / {:.0f} / {:.0f}".format(summary["ttft_ms"]["p50"], summary["ttft_ms"]["p90"], summary["ttft_ms"]["p99"]))
    print("itl  p50/p90/p99 ms: {:.1f} / {:.1f} / {:.1f}".format(summary["itl_ms"]["p50"], summary["itl_ms"]["p90"], summary["itl_ms"]["p99"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2)
        print("wrote", args.json)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
