# FastEmbed Ruby

[![Gem Version](https://badge.fury.io/rb/fastembed.svg)](https://rubygems.org/gems/fastembed)
[![CI](https://github.com/khasinski/fastembed-rb/actions/workflows/ci.yml/badge.svg)](https://github.com/khasinski/fastembed-rb/actions/workflows/ci.yml)

Fast, lightweight text embeddings in Ruby. Convert text into vectors for semantic search, similarity matching, clustering, and RAG applications.

```ruby
embedding = Fastembed::TextEmbedding.new
vectors = embedding.embed(["Hello world", "Ruby is great"]).to_a
# => [[0.123, -0.456, ...], [0.789, 0.012, ...]]  (384-dimensional vectors)
```

## What are embeddings?

Embeddings convert text into numerical vectors that capture semantic meaning. Similar texts produce similar vectors, enabling:

- **Semantic search** - Find relevant documents by meaning, not just keywords
- **Similarity matching** - Compare texts to find duplicates or related content
- **RAG applications** - Retrieve context for LLMs like ChatGPT
- **Clustering** - Group similar documents together

## Installation

```ruby
gem 'fastembed'
```

## Quick Start

```ruby
require 'fastembed'

# Create embedding model (downloads automatically on first use, ~67MB)
embedding = Fastembed::TextEmbedding.new

# Embed your texts
docs = ["Ruby is a programming language", "Python is also a programming language"]
vectors = embedding.embed(docs).to_a

# Find similarity between texts (cosine similarity via dot product)
similarity = vectors[0].zip(vectors[1]).sum { |a, b| a * b }
puts similarity  # => 0.89 (high similarity!)
```

## Semantic Search Example

```ruby
# Your document corpus
documents = [
  "The quick brown fox jumps over the lazy dog",
  "Machine learning is a subset of artificial intelligence",
  "Ruby on Rails is a web application framework",
  "Neural networks are inspired by biological brains"
]

# Create embeddings for all documents
embedding = Fastembed::TextEmbedding.new
doc_vectors = embedding.embed(documents).to_a

# Search query
query = "AI and deep learning"
query_vector = embedding.embed(query).first

# Find most similar document (highest dot product)
similarities = doc_vectors.map.with_index do |doc_vec, i|
  score = query_vector.zip(doc_vec).sum { |a, b| a * b }
  [i, score]
end

best_match = similarities.max_by { |_, score| score }
puts documents[best_match[0]]  # => "Machine learning is a subset of artificial intelligence"
```

## Usage

### Choose a Model

```ruby
# Default: Fast and accurate (384 dimensions, 67MB)
embedding = Fastembed::TextEmbedding.new

# Higher accuracy (768 dimensions, 210MB)
embedding = Fastembed::TextEmbedding.new(model_name: "BAAI/bge-base-en-v1.5")

# Multilingual support (100+ languages)
embedding = Fastembed::TextEmbedding.new(model_name: "intfloat/multilingual-e5-small")

# Long documents (8192 tokens vs default 512)
embedding = Fastembed::TextEmbedding.new(model_name: "nomic-ai/nomic-embed-text-v1.5")
```

### Process Large Datasets

```ruby
# Lazy evaluation - memory efficient for large datasets
documents = File.readlines("corpus.txt")

embedding.embed(documents, batch_size: 64).each_slice(100) do |batch|
  store_in_vector_database(batch)
end
```

### List Available Models

```ruby
Fastembed::TextEmbedding.list_supported_models.each do |model|
  puts "#{model[:model_name]} - #{model[:dim]}d - #{model[:description]}"
end
```

## Supported Models

| Model | Dim | Use Case |
|-------|-----|----------|
| `BAAI/bge-small-en-v1.5` | 384 | Default, fast English embeddings |
| `BAAI/bge-base-en-v1.5` | 768 | Higher accuracy English |
| `BAAI/bge-large-en-v1.5` | 1024 | Highest accuracy English |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | General purpose, lightweight |
| `sentence-transformers/all-mpnet-base-v2` | 768 | High quality general purpose |
| `intfloat/multilingual-e5-small` | 384 | 100+ languages |
| `intfloat/multilingual-e5-base` | 768 | 100+ languages, higher accuracy |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Long context (8192 tokens) |
| `jinaai/jina-embeddings-v2-base-en` | 768 | Long context (8192 tokens) |

## Performance

On Apple M1 Max with the default model:

| Batch Size | Throughput |
|------------|------------|
| 1 document | ~6.5ms |
| 100 documents | ~500 docs/sec |
| 1000 documents | ~500 docs/sec |

Larger models are slower but more accurate. See [benchmarks](BENCHMARKS.md) for details.

## Configuration

```ruby
Fastembed::TextEmbedding.new(
  model_name: "BAAI/bge-small-en-v1.5",  # Model to use
  cache_dir: "~/.cache/fastembed",        # Where to store models
  threads: 4,                              # ONNX Runtime threads
  providers: ["CUDAExecutionProvider"]     # GPU acceleration (Linux/Windows)
)
```

**Environment variables:**
- `FASTEMBED_CACHE_PATH` - Custom model cache directory

## Requirements

- Ruby >= 3.3
- ~70MB disk space for default model (varies by model)

## Acknowledgments

Ruby port of [FastEmbed](https://github.com/qdrant/fastembed) by Qdrant. Built on [onnxruntime-ruby](https://github.com/ankane/onnxruntime-ruby) and [tokenizers-ruby](https://github.com/ankane/tokenizers-ruby) by Andrew Kane.

## License

MIT
