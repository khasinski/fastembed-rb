# Benchmarks

Performance benchmarks on Apple M1 Max, Ruby 3.3, Python 3.13 (January 2026).

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

Comprehensive comparison of fastembed-rb against Python FastEmbed (v0.7.4) on Apple M1 Max.

### Text Embeddings (100 documents)

| Model | Ruby (docs/sec) | Python (docs/sec) | Ratio |
|-------|-----------------|-------------------|-------|
| BAAI/bge-small-en-v1.5 | 566 | 629 | 0.90x |
| BAAI/bge-base-en-v1.5 | 176 | 169 | **1.04x** |
| all-MiniLM-L6-v2 | 922 | 1309 | 0.70x |

Ruby is within 10-30% of Python for text embeddings. Both use the same ONNX Runtime backend.

### Rerankers (100 query-document pairs)

| Model | Ruby (pairs/sec) | Python (pairs/sec) | Ratio |
|-------|------------------|-------------------|-------|
| ms-marco-MiniLM-L-6-v2 | 986 | 982 | **1.00x** |
| ms-marco-MiniLM-L-12-v2 | 398 | 512 | 0.78x |
| BAAI/bge-reranker-base | 132 | 124 | **1.06x** |

Ruby matches or beats Python on rerankers.

### Sparse Embeddings - SPLADE (100 documents)

| Model | Ruby (docs/sec) | Python (docs/sec) | Ratio |
|-------|-----------------|-------------------|-------|
| Splade_PP_en_v1 | 23 | 108 | 0.21x |

Ruby's SPLADE implementation is slower due to post-processing overhead. Python uses optimized numpy operations for the log1p transformation.

### Late Interaction - ColBERT (100 documents)

| Model | Ruby (docs/sec) | Python (docs/sec) | Ratio |
|-------|-----------------|-------------------|-------|
| colbert-ir/colbertv2.0 | 191 | 184 | **1.04x** |

Ruby slightly outperforms Python for ColBERT embeddings.

### Image Embeddings (100 images)

| Model | Ruby (imgs/sec) | Python (imgs/sec) | Ratio |
|-------|-----------------|-------------------|-------|
| clip-ViT-B-32-vision | 9 | 42 | 0.22x |

Ruby's image embedding is slower due to MiniMagick subprocess overhead for image preprocessing. Python uses Pillow which is more efficient for batch processing.

### Summary

| Category | Ruby vs Python |
|----------|---------------|
| Text Embeddings | ~90% of Python speed |
| Rerankers | **Equal or faster** |
| ColBERT | **Equal or faster** |
| Sparse (SPLADE) | ~21% of Python speed |
| Image | ~22% of Python speed |

**Recommendation:** Ruby is excellent for text embeddings, reranking, and ColBERT. For heavy sparse or image embedding workloads, consider Python.

### Why the Differences?

Both implementations use the same ONNX Runtime for model inference. The differences come from:

1. **Text/Reranker/ColBERT** - Hot path is tokenization (Rust) + inference (C++). Minimal language overhead. Ruby matches Python.

2. **Sparse (SPLADE)** - Requires post-processing with log1p transformation. Python's numpy vectorization is faster than Ruby loops.

3. **Image** - Requires image preprocessing (resize, normalize). Python's Pillow is faster than Ruby's MiniMagick (subprocess-based).

### Memory Usage

| State | Ruby |
|-------|------|
| Initial | 33 MB |
| Model loaded | 277 MB |
| +1000 embeddings | 359 MB |
| After GC | 355 MB |

Memory is stable across multiple embedding rounds - no leaks detected.

### Embedding Quality

Both implementations produce identical embeddings (same ONNX models), verified by cosine similarity tests:

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

## Reranker Performance

TextCrossEncoder (cross-encoder) performance:

| Model | 100 pairs | Throughput |
|-------|-----------|------------|
| ms-marco-MiniLM-L-6-v2 | 102ms | **986 pairs/sec** |
| ms-marco-MiniLM-L-12-v2 | 252ms | **398 pairs/sec** |
| bge-reranker-base | 758ms | **132 pairs/sec** |

Cross-encoders are slower than embedding models because they process query-document pairs together rather than encoding them independently.

### Profiling Scripts

The `benchmark/` directory contains:

- `profile.rb` - Comprehensive embedding performance profiling
- `reranker_benchmark.rb` - Reranker/cross-encoder performance
- `memory_profile.rb` - Memory usage analysis
- `compare_python.py` - Python FastEmbed comparison
- `compare_all.rb` - Unified Ruby vs Python comparison

Run with:
```bash
ruby benchmark/profile.rb
ruby benchmark/reranker_benchmark.rb
ruby benchmark/memory_profile.rb
ruby benchmark/compare_all.rb
python3 benchmark/compare_python.py
```
