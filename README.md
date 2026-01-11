# FastEmbed Ruby

[![Gem Version](https://img.shields.io/gem/v/fastembed.svg)](https://rubygems.org/gems/fastembed)
[![CI](https://github.com/khasinski/fastembed-rb/actions/workflows/ci.yml/badge.svg)](https://github.com/khasinski/fastembed-rb/actions/workflows/ci.yml)

Fast, lightweight text embeddings in Ruby. A port of [FastEmbed](https://github.com/qdrant/fastembed) by Qdrant.

```ruby
embedding = Fastembed::TextEmbedding.new
vectors = embedding.embed(["The quick brown fox", "jumps over the lazy dog"]).to_a
```

Supports dense embeddings, sparse embeddings (SPLADE), late interaction (ColBERT), reranking, and image embeddings - all running locally with ONNX Runtime.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Text Embeddings](#text-embeddings)
- [Reranking](#reranking)
- [Sparse Embeddings](#sparse-embeddings)
- [Late Interaction (ColBERT)](#late-interaction-colbert)
- [Image Embeddings](#image-embeddings)
- [Async Processing](#async-processing)
- [Progress Tracking](#progress-tracking)
- [CLI](#cli)
- [Custom Models](#custom-models)
- [Configuration](#configuration)
- [Performance](#performance)

## Installation

Add to your Gemfile:

```ruby
gem "fastembed"
```

For image embeddings, also add:

```ruby
gem "mini_magick"
```

## Getting Started

```ruby
require "fastembed"

# Create an embedding model (downloads ~67MB on first use)
embedding = Fastembed::TextEmbedding.new

# Embed some text
documents = [
  "Ruby is a dynamic programming language",
  "Python is great for data science",
  "JavaScript runs in the browser"
]
vectors = embedding.embed(documents).to_a

# Each vector is 384 floats (for the default model)
vectors.first.length  # => 384
```

### Semantic Search

Find documents by meaning, not just keywords:

```ruby
embedding = Fastembed::TextEmbedding.new

# Your document corpus
documents = [
  "The cat sat on the mat",
  "Machine learning powers modern AI",
  "Ruby on Rails is a web framework",
  "Deep learning uses neural networks"
]
doc_vectors = embedding.embed(documents).to_a

# Search for a concept
query = "artificial intelligence and neural nets"
query_vector = embedding.embed([query]).first

# Find the most similar document (cosine similarity)
scores = doc_vectors.map { |v| query_vector.zip(v).sum { |a, b| a * b } }
best_idx = scores.each_with_index.max.last

puts documents[best_idx]  # => "Deep learning uses neural networks"
```

### Integration with Vector Databases

```ruby
# With Qdrant
require "qdrant"

embedding = Fastembed::TextEmbedding.new
client = Qdrant::Client.new(url: "http://localhost:6333")

# Index documents
documents.each_with_index do |doc, i|
  vector = embedding.embed([doc]).first
  client.points.upsert(
    collection_name: "docs",
    points: [{ id: i, vector: vector, payload: { text: doc } }]
  )
end

# Search
query_vector = embedding.embed(["your search query"]).first
results = client.points.search(collection_name: "docs", vector: query_vector, limit: 5)
```

## Text Embeddings

### Choose a Model

```ruby
# Default: fast and accurate (384 dimensions, 67MB)
embedding = Fastembed::TextEmbedding.new

# Higher accuracy (768 dimensions, 210MB)
embedding = Fastembed::TextEmbedding.new(model_name: "BAAI/bge-base-en-v1.5")

# Multilingual - 100+ languages (384 dimensions)
embedding = Fastembed::TextEmbedding.new(model_name: "intfloat/multilingual-e5-small")

# Long documents - 8192 token context (768 dimensions)
embedding = Fastembed::TextEmbedding.new(model_name: "nomic-ai/nomic-embed-text-v1.5")
```

### Supported Models

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | 67MB | Default, fast |
| `BAAI/bge-base-en-v1.5` | 768 | 210MB | Better accuracy |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.2GB | Best accuracy |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90MB | General purpose |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 440MB | High quality |
| `intfloat/multilingual-e5-small` | 384 | 450MB | 100+ languages |
| `intfloat/multilingual-e5-base` | 768 | 1.1GB | Multilingual, better |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 520MB | 8192 token context |
| `jinaai/jina-embeddings-v2-base-en` | 768 | 520MB | 8192 token context |

### Query vs Passage Embeddings

For asymmetric search (short queries, long documents), use specialized methods:

```ruby
# For search queries
query_vectors = embedding.query_embed(["What is Ruby?"]).to_a

# For documents/passages
doc_vectors = embedding.passage_embed(documents).to_a
```

### Lazy Evaluation

Embeddings are generated lazily, making it memory-efficient for large datasets:

```ruby
# Process millions of documents without loading all vectors into memory
File.foreach("documents.txt").lazy.each_slice(1000) do |batch|
  embedding.embed(batch).each do |vector|
    store_in_database(vector)
  end
end
```

## Reranking

Rerankers score query-document pairs for more accurate relevance ranking. Use them after initial retrieval:

```ruby
reranker = Fastembed::TextCrossEncoder.new

query = "What is machine learning?"
documents = [
  "Machine learning is a branch of AI",
  "The weather is nice today",
  "Deep learning uses neural networks"
]

# Get raw scores (higher = more relevant)
scores = reranker.rerank(query: query, documents: documents)
# => [8.5, -10.2, 5.3]

# Get sorted results with metadata
results = reranker.rerank_with_scores(query: query, documents: documents, top_k: 2)
# => [
#   { document: "Machine learning is...", score: 8.5, index: 0 },
#   { document: "Deep learning uses...", score: 5.3, index: 2 }
# ]
```

### Reranker Models

| Model | Size | Notes |
|-------|------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB | Default, fast |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 120MB | Better accuracy |
| `BAAI/bge-reranker-base` | 1.1GB | High accuracy |
| `BAAI/bge-reranker-large` | 2.2GB | Best accuracy |

## Sparse Embeddings

SPLADE models produce sparse vectors where each dimension corresponds to a vocabulary term. Great for hybrid search:

```ruby
sparse = Fastembed::TextSparseEmbedding.new

result = sparse.embed(["Ruby programming language"]).first
# => #<SparseEmbedding indices=[1234, 5678, ...] values=[0.8, 1.2, ...]>

result.indices  # vocabulary token IDs with non-zero weights
result.values   # corresponding weights
result.nnz      # number of non-zero elements
```

### Hybrid Search

Combine dense and sparse embeddings for better results:

```ruby
dense = Fastembed::TextEmbedding.new
sparse = Fastembed::TextSparseEmbedding.new

documents = ["your documents here"]

# Generate both types of embeddings
dense_vectors = dense.embed(documents).to_a
sparse_vectors = sparse.embed(documents).to_a

# Store both in your vector database and combine scores at query time
```

## Late Interaction (ColBERT)

ColBERT produces token-level embeddings for fine-grained matching:

```ruby
colbert = Fastembed::LateInteractionTextEmbedding.new

query = colbert.query_embed(["What is Ruby?"]).first
doc = colbert.embed(["Ruby is a programming language"]).first

# MaxSim scoring - sum of max similarities per query token
score = query.max_sim(doc)
```

### Late Interaction Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `colbert-ir/colbertv2.0` | 128 | Default |
| `jinaai/jina-colbert-v1-en` | 768 | 8192 token context |

## Image Embeddings

Convert images to vectors for visual search:

```ruby
# Requires mini_magick gem
image_embed = Fastembed::ImageEmbedding.new

# From file paths
vectors = image_embed.embed(["photo1.jpg", "photo2.png"]).to_a

# From URLs
vectors = image_embed.embed(["https://example.com/image.jpg"]).to_a
```

### Image Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `Qdrant/clip-ViT-B-32-vision` | 512 | Default, CLIP |
| `Qdrant/resnet50-onnx` | 2048 | ResNet50 |
| `jinaai/jina-clip-v1` | 768 | Jina CLIP |

## Async Processing

Run embeddings in background threads:

```ruby
embedding = Fastembed::TextEmbedding.new

# Start async embedding
future = embedding.embed_async(large_document_list)

# Do other work...

# Get results when ready (blocks until complete)
vectors = future.value
```

### Parallel Processing

```ruby
# Process multiple batches concurrently
futures = documents.each_slice(1000).map do |batch|
  embedding.embed_async(batch)
end

# Wait for all and combine results
all_vectors = futures.flat_map(&:value)
```

### Future Methods

```ruby
future.complete?  # check if done
future.pending?   # check if still running
future.success?   # completed without error?
future.failure?   # completed with error?
future.error      # get the error if failed
future.wait(timeout: 5)  # wait up to 5 seconds

# Chaining
future.then { |vectors| vectors.map(&:first) }
      .rescue { |e| puts "Error: #{e}" }
```

### Async Utilities

```ruby
# Wait for all futures
results = Fastembed::Async.all(futures)

# Get first completed result
result = Fastembed::Async.race(futures, timeout: 10)
```

## Progress Tracking

Track progress for large embedding jobs:

```ruby
embedding = Fastembed::TextEmbedding.new

documents = Array.new(10_000) { "document text" }

embedding.embed(documents, batch_size: 256) do |progress|
  puts "Batch #{progress.current}/#{progress.total}"
  puts "#{(progress.percentage * 100).round}% complete"
  puts "~#{progress.documents_processed} documents processed"
end.to_a
```

## CLI

FastEmbed includes a command-line tool:

```bash
# List available models
fastembed list           # embedding models
fastembed list-reranker  # reranker models
fastembed list-sparse    # sparse models
fastembed list-image     # image models

# Get model info
fastembed info "BAAI/bge-small-en-v1.5"

# Pre-download a model
fastembed download "BAAI/bge-base-en-v1.5"

# Embed text (outputs JSON)
fastembed embed "Hello world" "Another text"

# Different output formats
fastembed embed -f ndjson "Hello world"
fastembed embed -f csv "Hello world"

# Read from file
fastembed embed -i documents.txt

# Use different model
fastembed embed -m "BAAI/bge-base-en-v1.5" "Hello"

# Rerank documents
fastembed rerank "query" "doc1" "doc2" "doc3"

# Benchmark a model
fastembed benchmark -m "BAAI/bge-small-en-v1.5" -n 100
```

## Custom Models

Register custom models from HuggingFace:

```ruby
Fastembed.register_model(
  model_name: "my-org/my-model",
  dim: 768,
  description: "My custom model",
  sources: { hf: "my-org/my-model" },
  model_file: "onnx/model.onnx"
)

# Now use it like any other model
embedding = Fastembed::TextEmbedding.new(model_name: "my-org/my-model")
```

### Load from Local Directory

```ruby
embedding = Fastembed::TextEmbedding.new(
  local_model_dir: "/path/to/model",
  model_file: "model.onnx",
  tokenizer_file: "tokenizer.json"
)
```

## Configuration

### Initialization Options

```ruby
Fastembed::TextEmbedding.new(
  model_name: "BAAI/bge-small-en-v1.5",  # model to use
  cache_dir: "~/.cache/fastembed",        # where to store models
  threads: 4,                              # ONNX Runtime threads
  providers: ["CUDAExecutionProvider"],    # GPU acceleration
  show_progress: true,                     # show download progress
  quantization: :q4                        # use quantized model
)
```

### Quantization

Use smaller, faster models with quantization:

```ruby
# Available: :fp32 (default), :fp16, :int8, :uint8, :q4
embedding = Fastembed::TextEmbedding.new(quantization: :int8)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FASTEMBED_CACHE_PATH` | Custom model cache directory |
| `HF_TOKEN` | HuggingFace token for private models |

### GPU Acceleration

```ruby
# CUDA (Linux/Windows with NVIDIA GPU)
embedding = Fastembed::TextEmbedding.new(
  providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# CoreML (macOS)
embedding = Fastembed::TextEmbedding.new(
  providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
```

## Performance

On Apple M1 Max with the default model (BAAI/bge-small-en-v1.5):

| Batch Size | Documents/sec | Latency |
|------------|--------------|---------|
| 1 | ~150 | ~6.5ms |
| 32 | ~500 | ~64ms |
| 256 | ~550 | ~465ms |

Larger models are slower but more accurate. See [BENCHMARKS.md](BENCHMARKS.md) for detailed comparisons.

## Requirements

- Ruby >= 3.3
- ~70MB-2GB disk space (varies by model)

## Acknowledgments

Ruby port of [FastEmbed](https://github.com/qdrant/fastembed) by Qdrant. Built on [onnxruntime-ruby](https://github.com/ankane/onnxruntime-ruby) and [tokenizers-ruby](https://github.com/ankane/tokenizers-ruby) by Andrew Kane.

## License

MIT
