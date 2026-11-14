#!/usr/bin/env python3
"""
scripts/benchmark_agent.py

Empirical Concurrency & Throughput Benchmark for KnowVex Agent Queries.
Measures Time-To-First-Token (TTFT), token streaming rates, and concurrency
saturation on direct_answer / general chat responses.

Usage:
    # 1. Quick test with 10 concurrent queries:
    python scripts/benchmark_agent.py --concurrency 10 --queries 10

    # 2. Ramp test with 30 concurrent users:
    python scripts/benchmark_agent.py --concurrency 30 --queries 60

    # 3. Target remote or custom server:
    python scripts/benchmark_agent.py --base-url "http://localhost:8000" --concurrency 25
"""

import sys
import time
import math
import asyncio
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
from backend.services.security import create_access_token


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile from a sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[int(f)] * (c - k) + data[int(c)] * (k - f)


async def get_or_create_benchmark_token(client: httpx.AsyncClient, base_url: str) -> str:
    """Generate a clean test token for benchmarking."""
    # We create a token directly using our security primitives with User ID 1
    # This avoids hitting authentication rate limits or email services.
    return create_access_token(user_id="1", role="admin")


async def get_or_create_benchmark_session(client: httpx.AsyncClient, base_url: str, token: str) -> str:
    """Create a temporary chat session for the benchmark."""
    headers = {"Authorization": f"Bearer {token}"}
    res = await client.post(f"{base_url}/api/v1/sessions/", headers=headers, json={"title": "Benchmark Session"})
    if res.status_code == 201:
        data = res.json()
        return data.get("session_id") or data.get("id")
    
    # Fallback: check if we can query existing sessions
    list_res = await client.get(f"{base_url}/api/v1/sessions/", headers=headers)
    if list_res.status_code == 200:
        sessions = list_res.json().get("sessions", [])
        if sessions:
            return sessions[0].get("session_id") or sessions[0].get("id")
    
    raise RuntimeError(f"Could not initialize benchmark session: HTTP {res.status_code} - {res.text}")


async def run_single_streaming_query(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    session_id: str,
    prompt: str,
    query_idx: int,
) -> Dict[str, Any]:
    """Execute a single streaming chat query and record timing metrics."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
    }
    payload = {
        "question": prompt,
        "mode": "auto",
    }

    t_start = time.perf_counter()
    first_token_time: float | None = None
    token_chunks = 0
    total_bytes = 0
    status_code = 0
    error_msg = None

    try:
        async with client.stream(
            "POST",
            f"{base_url}/api/v1/chat/{session_id}",
            json=payload,
            headers=headers,
            timeout=45.0,
        ) as response:
            status_code = response.status_code
            if status_code != 200:
                error_msg = f"HTTP {status_code}"
                return {
                    "query_idx": query_idx,
                    "status_code": status_code,
                    "ttft_ms": None,
                    "total_ms": (time.perf_counter() - t_start) * 1000,
                    "tokens": 0,
                    "tokens_per_sec": 0,
                    "error": error_msg,
                }

            async for line in response.aiter_lines():
                if not line:
                    continue
                total_bytes += len(line)
                if line.startswith("data:"):
                    raw_data = line[5:].strip()
                    if raw_data and raw_data != "[DONE]":
                        if first_token_time is None:
                            first_token_time = (time.perf_counter() - t_start) * 1000
                        token_chunks += 1

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = first_token_time if first_token_time is not None else total_ms
        tps = (token_chunks / (total_ms / 1000)) if total_ms > 0 else 0

        return {
            "query_idx": query_idx,
            "status_code": status_code,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "tokens": token_chunks,
            "tokens_per_sec": tps,
            "error": None,
        }

    except Exception as exc:
        return {
            "query_idx": query_idx,
            "status_code": status_code or 500,
            "ttft_ms": None,
            "total_ms": (time.perf_counter() - t_start) * 1000,
            "tokens": 0,
            "tokens_per_sec": 0,
            "error": str(exc),
        }


async def benchmark_concurrency(
    base_url: str,
    concurrency: int,
    total_queries: int,
    prompt: str,
) -> None:
    """Run concurrent benchmark worker pool and print detailed percentile analysis."""
    print("\n" + "═" * 70)
    print(f"🚀 KnowVex Agent Concurrency & TTFT Benchmark")
    print(f"   • Target Server:      {base_url}")
    print(f"   • Concurrency Level:  {concurrency} simultaneous queries")
    print(f"   • Total Requests:     {total_queries}")
    print(f"   • Test Query:         \"{prompt}\"")
    print("═" * 70 + "\n")

    limits = httpx.Limits(max_keepalive_connections=concurrency + 10, max_connections=concurrency + 20)
    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        # Step 1: Health check
        try:
            health = await client.get(f"{base_url}/health")
            if health.status_code != 200:
                print(f"❌ Server health check failed: HTTP {health.status_code}")
                return
            print("✅ Server reachable and healthy.")
        except Exception as e:
            print(f"❌ Cannot connect to server at {base_url}: {e}")
            print("   Make sure the backend is running (`fastapi dev main.py`)")
            return

        # Step 2: Session and Auth setup
        token = await get_or_create_benchmark_token(client, base_url)
        session_id = await get_or_create_benchmark_session(client, base_url, token)
        print(f"✅ Benchmark session initialized: {session_id}")

        # Step 3: Run queries with concurrency semaphore
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict[str, Any]] = []

        async def worker(idx: int):
            async with semaphore:
                res = await run_single_streaming_query(client, base_url, token, session_id, prompt, idx)
                results.append(res)
                # Print live ticker
                completed = len(results)
                status_icon = "🟢" if res["error"] is None else "🔴"
                ttft_str = f"{res['ttft_ms']:.1f}ms" if res["ttft_ms"] else "FAILED"
                print(f"  [{completed:02d}/{total_queries:02d}] {status_icon} Query #{idx:02d} | TTFT: {ttft_str} | Status: {res['status_code']}")

        print(f"\n⚡ Firing {total_queries} queries with {concurrency} workers...\n")
        bench_start = time.perf_counter()
        tasks = [worker(i + 1) for i in range(total_queries)]
        await asyncio.gather(*tasks)
        bench_end = time.perf_counter()

    bench_duration = bench_end - bench_start

    # Step 4: Analytics and Report
    successful = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    ttfts = sorted([r["ttft_ms"] for r in successful if r["ttft_ms"] is not None])
    totals = sorted([r["total_ms"] for r in successful])
    tps_list = [r["tokens_per_sec"] for r in successful]
    total_tokens = sum(r["tokens"] for r in successful)

    print("\n" + "═" * 70)
    print("📊 BENCHMARK RESULTS & CAPACITY REPORT")
    print("═" * 70)
    print(f"   • Total Time Taken:         {bench_duration:.2f} s")
    print(f"   • Overall Throughput:       {len(successful) / bench_duration:.2f} queries/sec")
    print(f"   • Success Rate:             {len(successful)}/{total_queries} ({len(successful)/total_queries*100:.1f}%)")
    print(f"   • Failed Requests:          {len(failed)}")
    if failed:
        error_counts: Dict[str, int] = {}
        for f in failed:
            err = f["error"] or "Unknown"
            error_counts[err] = error_counts.get(err, 0) + 1
        for err, cnt in error_counts.items():
            print(f"     └─ {err}: {cnt} occurrences")

    if successful:
        print("\n⏱️  TIME TO FIRST TOKEN (TTFT) LATENCY:")
        print(f"   • Min:                      {min(ttfts):.1f} ms")
        print(f"   • Median (P50):             {percentile(ttfts, 50):.1f} ms")
        print(f"   • P90:                      {percentile(ttfts, 90):.1f} ms")
        print(f"   • P95:                      {percentile(ttfts, 95):.1f} ms")
        print(f"   • P99:                      {percentile(ttfts, 99):.1f} ms")
        print(f"   • Max:                      {max(ttfts):.1f} ms")

        print("\n⏳ TOTAL STREAM COMPLETION TIME:")
        print(f"   • Median (P50):             {percentile(totals, 50):.1f} ms")
        print(f"   • P95:                      {percentile(totals, 95):.1f} ms")

        print("\n⚡ STREAMING TOKEN THROUGHPUT:")
        print(f"   • Total Tokens Streamed:    {total_tokens} tokens")
        print(f"   • Aggregate Token Rate:     {total_tokens / bench_duration:.1f} tokens/sec")
        print(f"   • Average Per-Stream Rate:  {sum(tps_list) / len(tps_list):.1f} tokens/sec")

    print("═" * 70)

    # Diagnostic Insight
    if successful and percentile(ttfts, 90) < 500:
        print("\n💡 VERDICT: EXCELLENT. Pure async FastAPI is handling this concurrency with sub-500ms TTFT.")
        print("   No worker queue is needed for chat queries at this concurrency level.")
    elif successful and percentile(ttfts, 90) < 1500:
        print("\n💡 VERDICT: ACCEPTABLE. Latency is slightly elevated under load, but manageable.")
        print("   Check database pool size and upstream LLM rate limits if scaling further.")
    else:
        print("\n⚠️ VERDICT: BOTTLENECK DETECTED. TTFT exceeded 1.5s or requests failed.")
        print("   Primary suspects: 1) Upstream LLM provider 429 rate limits, 2) DB connection pool exhaustion.")
    print("═" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="KnowVex Agent Concurrency & Streaming Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of KnowVex API")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Number of concurrent query workers (default: 10)")
    parser.add_argument("--queries", "-q", type=int, default=10, help="Total number of queries to execute (default: 10)")
    parser.add_argument("--prompt", "-p", default="Explain what a binary search tree is in two concise sentences.", help="Test prompt to send")

    args = parser.parse_args()
    try:
        asyncio.run(benchmark_concurrency(args.base_url, args.concurrency, args.queries, args.prompt))
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user.")


if __name__ == "__main__":
    main()
