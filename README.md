# FastEmbed Ruby

A Ruby port of [FastEmbed](https://github.com/qdrant/fastembed) - a lightweight, fast library for generating text embeddings using ONNX Runtime.

## Features

- Fast text embeddings using ONNX Runtime
- Automatic model downloading and caching from HuggingFace
- Memory-efficient lazy evaluation with Enumerator
- Multiple pre-trained models supported
- No PyTorch dependency - lightweight and serverless-friendly

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'fastembed'
```

And then execute:

```bash
bundle install
```

Or install it yourself as:

```bash
gem install fastembed
```

## Usage

### Basic Usage

```ruby
require 'fastembed'

# Create an embedding model (downloads on first use)
embedding = Fastembed::TextEmbedding.new

# Generate embeddings
documents = [
  "This is a test document",
  "Another document to embed"
]

vectors = embedding.embed(documents).to_a
# => [[0.123, -0.456, ...], [0.789, -0.012, ...]]

# Each vector has 384 dimensions (for the default model)
puts vectors.first.length  # => 384
```

### Custom Model

```ruby
# Use a larger model for higher accuracy
embedding = Fastembed::TextEmbedding.new(
  model_name: "BAAI/bge-base-en-v1.5"
)

# Get embedding dimension
puts embedding.dim  # => 768
```

### Lazy Evaluation for Large Datasets

```ruby
# Process embeddings lazily to save memory
documents = File.readlines("large_corpus.txt")

embedding.embed(documents).each_slice(100) do |batch|
  # Process batch of 100 vectors at a time
  store_in_database(batch)
end
```

### Query and Passage Embeddings

For retrieval tasks, use prefixed embeddings:

```ruby
# Embed queries (optimized for similarity search)
query_vectors = embedding.query_embed(["What is machine learning?"]).to_a

# Embed passages (optimized for being searched)
passage_vectors = embedding.passage_embed(["Machine learning is..."]).to_a
```

### List Supported Models

```ruby
Fastembed::TextEmbedding.list_supported_models.each do |model|
  puts "#{model[:model_name]} - #{model[:dim]} dimensions"
end
```

## Supported Models

| Model | Dimensions | Description |
|-------|-----------|-------------|
| BAAI/bge-small-en-v1.5 | 384 | Fast and accurate English (default) |
| BAAI/bge-base-en-v1.5 | 768 | Balanced accuracy and speed |
| BAAI/bge-large-en-v1.5 | 1024 | Highest accuracy English |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Lightweight general-purpose |
| intfloat/multilingual-e5-small | 384 | 100+ languages support |
| intfloat/multilingual-e5-base | 768 | Larger multilingual model |

## Benchmarks

Performance benchmarks on Apple M1 Max, Ruby 3.3.10, using the default model (BAAI/bge-small-en-v1.5):

### Single Document Latency

| Text Length | Latency |
|-------------|---------|
| Short (~10 tokens) | ~6.5 ms |
| Medium (~30 tokens) | ~6.5 ms |

### Batch Throughput (100 documents)

| Text Length | Time | Throughput |
|-------------|------|------------|
| Short sentences | 0.2s | **502 docs/sec** |
| Medium paragraphs | 0.5s | **197 docs/sec** |
| Long documents | 2.3s | **44 docs/sec** |

### Large Scale

| Documents | Time | Throughput |
|-----------|------|------------|
| 1,000 short texts | 2.0s | **509 docs/sec** |

### Model Comparison

| Model | Dimensions | Size | Throughput |
|-------|-----------|------|------------|
| bge-small-en-v1.5 | 384 | 67 MB | **530 docs/sec** |
| bge-base-en-v1.5 | 768 | 210 MB | **169 docs/sec** |
| bge-large-en-v1.5 | 1024 | 1.2 GB | **50 docs/sec** |

### CPU vs CoreML (Apple Silicon)

| Model | CPU | CoreML | Winner |
|-------|-----|--------|--------|
| bge-small | 530 docs/sec | 131 docs/sec | CPU (4x faster) |
| bge-base | 169 docs/sec | 69 docs/sec | CPU (2.5x faster) |
| bge-large | 50 docs/sec | 16 docs/sec | CPU (3x faster) |

> **Note:** On Apple Silicon, the CPU provider consistently outperforms CoreML for embedding models. The ONNX Runtime CPU implementation is highly optimized for M1/M2 chips. Stick with the default CPU provider.

## Configuration

### Custom Cache Directory

```ruby
# Set via environment variable
ENV['FASTEMBED_CACHE_PATH'] = '/path/to/cache'

# Or via constructor
embedding = Fastembed::TextEmbedding.new(cache_dir: '/path/to/cache')
```

### Threading

```ruby
# Control ONNX Runtime threads
embedding = Fastembed::TextEmbedding.new(threads: 4)
```

### Execution Providers

```ruby
# Use CoreML on macOS (not recommended for small models)
embedding = Fastembed::TextEmbedding.new(
  providers: ["CoreMLExecutionProvider"]
)

# Use CUDA on Linux/Windows
embedding = Fastembed::TextEmbedding.new(
  providers: ["CUDAExecutionProvider"]
)
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests.

```bash
bundle install
bundle exec rspec
```

## Contributing

Bug reports and pull requests are welcome on GitHub.

## License

The gem is available as open source under the terms of the MIT License.

## Acknowledgments

- [FastEmbed](https://github.com/qdrant/fastembed) - Original Python implementation by Qdrant
- [onnxruntime-ruby](https://github.com/ankane/onnxruntime-ruby) - ONNX Runtime bindings for Ruby
- [tokenizers-ruby](https://github.com/ankane/tokenizers-ruby) - HuggingFace Tokenizers for Ruby
