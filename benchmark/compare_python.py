#!/usr/bin/env python3
"""Compare Python FastEmbed performance with Ruby FastEmbed."""

import time
from fastembed import TextEmbedding

TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Ruby on Rails is a server-side web application framework written in Ruby under the MIT License.",
    "Vector databases store embeddings and enable fast similarity search across millions of documents.",
    "Natural language processing helps computers understand, interpret, and generate human language.",
    "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing tests."
]

def benchmark_python():
    print("=" * 60)
    print("PYTHON FASTEMBED BENCHMARK")
    print("=" * 60)
    print()

    # Model loading time
    start = time.time()
    embedding = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    load_time = time.time() - start
    print(f"Model load time: {load_time * 1000:.1f}ms")

    # Warmup
    list(embedding.embed(["warmup"]))

    # Single document latency
    print()
    print("Single document latency:")
    for i, text in enumerate(TEXTS[:3]):
        times = []
        for _ in range(10):
            start = time.time()
            list(embedding.embed([text]))
            times.append(time.time() - start)
        avg = sum(times) / len(times)
        min_time = min(times)
        print(f"  Text {i+1} ({len(text)} chars): avg {avg*1000:.2f}ms, min {min_time*1000:.2f}ms")

    # Throughput
    print()
    print("Throughput:")
    for count in [100, 500, 1000]:
        texts = (TEXTS * (count // len(TEXTS) + 1))[:count]

        start = time.time()
        list(embedding.embed(texts, batch_size=64))
        elapsed = time.time() - start

        rate = count / elapsed
        print(f"  {count} docs: {rate:.1f} docs/sec ({elapsed*1000:.1f}ms)")

    print()
    print("=" * 60)

if __name__ == "__main__":
    benchmark_python()
