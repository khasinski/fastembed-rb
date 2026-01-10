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

## Ruby vs Python FastEmbed

We compared fastembed-rb against the original Python FastEmbed (v0.7.4) on Apple M1 Max.

### Performance Comparison

| Metric | Ruby | Python | Winner |
|--------|------|--------|--------|
| Model load time | **52-95ms** | 150ms | Ruby 1.6-2.9x |
| Single doc latency | **2.3ms** | 4.5ms | Ruby 2x |
| Throughput (100 docs) | 288/sec | 243/sec | Ruby 1.2x |
| Throughput (500 docs) | 287/sec | 260/sec | Ruby 1.1x |
| Throughput (1000 docs) | 288/sec | **317/sec** | Python 1.1x |

### Why is Ruby Competitive?

Both implementations use the same underlying technology:

1. **Same ONNX Runtime** - Both use ONNX Runtime for model inference. The actual neural network computation is identical C++ code.

2. **Same tokenizer** - Both use HuggingFace Tokenizers (Rust-based). Ruby uses `tokenizers-ruby`, Python uses `tokenizers` - same Rust core.

3. **Minimal language overhead** - The hot path (tokenization + inference) happens in native code. Ruby/Python are just orchestrating.

Ruby's advantages:
- **Simpler architecture** - fastembed-rb is ~500 lines of Ruby. Python FastEmbed has more abstraction layers.
- **Less overhead** - Ruby's C extensions have efficient FFI. Python's numpy array conversions add overhead.
- **Faster model loading** - Ruby's ONNX binding initializes faster.

Python's advantages:
- **Better batching at scale** - Python's numpy enables more efficient large batch operations.
- **More mature optimization** - Years of tuning for ML workloads.

### Memory Usage

| State | Ruby |
|-------|------|
| Initial | 33 MB |
| Model loaded | 277 MB |
| +1000 embeddings | 359 MB |
| After GC | 355 MB |

Memory is stable across multiple embedding rounds - no leaks detected.

### Embedding Quality

Both implementations produce identical embeddings (same ONNX model), verified by cosine similarity tests:

```
'dog' vs 'puppy' = 0.855 (high - PASS)
'dog' vs 'cat' = 0.688 (medium - PASS)
'machine learning' vs 'artificial intelligence' = 0.718 (high - PASS)
'machine learning' vs 'cooking recipes' = 0.426 (low - PASS)
```

## Running Your Own Benchmarks

```ruby
require 'fastembed'
require 'benchmark'

embedding = Fastembed::TextEmbedding.new
texts = Array.new(1000) { "Sample text for benchmarking" }

result = Benchmark.measure { embedding.embed(texts).to_a }
puts "#{1000 / result.real} docs/sec"
```

### Profiling Scripts

The `benchmark/` directory contains:

- `profile.rb` - Comprehensive performance profiling
- `memory_profile.rb` - Memory usage analysis
- `compare_python.py` - Python FastEmbed comparison

Run with:
```bash
ruby benchmark/profile.rb
ruby benchmark/memory_profile.rb
python3 benchmark/compare_python.py
```
