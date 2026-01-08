# Benchmarks

Performance benchmarks on Apple M1 Max, Ruby 3.3.10.

## Single Document Latency

Using the default model (BAAI/bge-small-en-v1.5):

| Text Length | Latency |
|-------------|---------|
| Short (~10 tokens) | ~6.5 ms |
| Medium (~30 tokens) | ~6.5 ms |

## Batch Throughput

### bge-small-en-v1.5 (default)

| Text Length | 100 docs | Throughput |
|-------------|----------|------------|
| Short sentences | 0.2s | **502 docs/sec** |
| Medium paragraphs | 0.5s | **197 docs/sec** |
| Long documents | 2.3s | **44 docs/sec** |

### Large Scale (1000 documents)

| Model | Time | Throughput |
|-------|------|------------|
| bge-small-en-v1.5 | 2.0s | **509 docs/sec** |

## Model Comparison

| Model | Dimensions | Size | Throughput |
|-------|-----------|------|------------|
| bge-small-en-v1.5 | 384 | 67 MB | **530 docs/sec** |
| bge-base-en-v1.5 | 768 | 210 MB | **169 docs/sec** |
| bge-large-en-v1.5 | 1024 | 1.2 GB | **50 docs/sec** |

## CPU vs CoreML (Apple Silicon)

We tested CoreML execution provider to see if GPU/Neural Engine acceleration helps.

### Results by Model

| Model | CPU | CoreML (best batch) | Ratio |
|-------|-----|---------------------|-------|
| bge-small | 418/s | 162/s (batch=64) | CPU 2.6x faster |
| bge-base | 134/s | 64/s (batch=32) | CPU 2.1x faster |
| bge-large | 41/s | 23/s (batch=16) | CPU 1.8x faster |

### Batch Size Impact on CoreML

| Batch Size | CPU (bge-small) | CoreML | Ratio |
|------------|-----------------|--------|-------|
| 1 | 209/s | 40/s | 0.19x |
| 8 | 351/s | 118/s | 0.34x |
| 16 | 382/s | 146/s | 0.38x |
| 32 | 410/s | 155/s | 0.38x |
| 64 | 418/s | 162/s | 0.39x |
| 128 | 391/s | 139/s | 0.35x |
| 256 | 412/s | 121/s | 0.29x |

### Conclusion

**CPU is faster than CoreML** for all embedding models on Apple Silicon. The gap narrows for larger models, but CPU still wins. This is because:

1. ONNX Runtime's CPU implementation is highly optimized for M1/M2
2. Data transfer overhead to Neural Engine outweighs compute benefits
3. Embedding models are relatively small compared to LLMs

**Recommendation:** Stick with the default CPU provider.

## Running Your Own Benchmarks

```ruby
require 'fastembed'
require 'benchmark'

embedding = Fastembed::TextEmbedding.new
texts = Array.new(1000) { "Sample text for benchmarking" }

result = Benchmark.measure { embedding.embed(texts).to_a }
puts "#{1000 / result.real} docs/sec"
```
