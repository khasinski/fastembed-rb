# FastEmbed-rb Roadmap

This document outlines features from the original [FastEmbed Python library](https://github.com/qdrant/fastembed) that are not yet implemented in fastembed-rb.

## Current Status (v1.0.0)

### Implemented
- Dense text embeddings with 12 models
- Automatic model downloading from HuggingFace
- Lazy evaluation via `Enumerator`
- Query/passage prefixes for retrieval models
- Mean pooling and L2 normalization
- Configurable batch size and threading
- CoreML execution provider support
- CLI tool (`fastembed`)
- **Reranking / Cross-Encoder models** (5 models)

## Feature Gap Analysis

### High Priority

#### 1. Sparse Text Embeddings
The Python library supports sparse embedding models that return indices and values rather than dense vectors. These are useful for hybrid search combining keyword and semantic matching.

**Models to support:**
- `Qdrant/bm25` - Classic BM25 (0.010 GB)
- `Qdrant/bm42-all-minilm-l6-v2-attentions` - Attention-based sparse (0.090 GB)
- `prithivida/Splade_PP_en_v1` - SPLADE++ (0.532 GB)

**API design:**
```ruby
sparse = Fastembed::SparseTextEmbedding.new
result = sparse.embed(["hello world"]).first
# => { indices: [123, 456, 789], values: [0.5, 0.3, 0.2] }
```

**Implementation notes:**
- Need new `SparseTextEmbedding` class
- Different output format (sparse vectors instead of dense)
- May require different tokenization approach for BM25

#### 2. Late Interaction (ColBERT) Models
ColBERT-style models produce token-level embeddings rather than a single vector per document. This enables more fine-grained matching.

**Models to support:**
- `answerdotai/answerai-colbert-small-v1` (96 dim)
- `colbert-ir/colbertv2.0` (128 dim)
- `jinaai/jina-colbert-v2` (128 dim)

**API design:**
```ruby
colbert = Fastembed::LateInteractionTextEmbedding.new
result = colbert.embed(["hello world"]).first
# => Array of token embeddings, shape: [num_tokens, dim]
```

**Implementation notes:**
- Returns 2D array per document (tokens × dimensions)
- Different pooling strategy (no pooling, keep all tokens)
- Scoring requires MaxSim operation between query and document tokens

#### ~~3. Reranking / Cross-Encoder Models~~ ✅ IMPLEMENTED

See `Fastembed::TextCrossEncoder` class.

### Medium Priority

#### 4. Image Embeddings
Vision models for converting images to vectors. Useful for image search and multimodal applications.

**Models to support:**
- `Qdrant/resnet50-onnx` (2048 dim)
- `Qdrant/clip-ViT-B-32-vision` (512 dim)
- `jinaai/jina-clip-v1` (768 dim)

**API design:**
```ruby
image_embed = Fastembed::ImageEmbedding.new
vector = image_embed.embed(["path/to/image.jpg"]).first
# => [0.1, 0.2, ...]
```

**Implementation notes:**
- Requires image preprocessing (resize, normalize)
- May need `mini_magick` or `vips` gem for image loading
- CLIP models can embed both images and text into same space

#### 5. Custom Model Support
Allow users to load arbitrary ONNX models not in the registry.

**API design:**
```ruby
Fastembed.add_custom_model(
  model_name: "my-org/my-model",
  dim: 768,
  sources: { hf: "my-org/my-model" },
  model_file: "model.onnx"
)

embed = Fastembed::TextEmbedding.new(model_name: "my-org/my-model")
```

**Implementation notes:**
- Validate ONNX model structure
- Allow custom pooling strategies
- Support local file paths in addition to HuggingFace

### Low Priority

#### 6. Multimodal Late Interaction (ColPali)
ColPali models that can embed both images and text for document retrieval.

**Models to support:**
- `vidore/colpali-v1.2`
- `vidore/colqwen2-v1.0`

**Implementation notes:**
- Combines image and text embedding
- Requires vision preprocessing
- Complex architecture, lower priority

#### 7. Quantized Models
Support for INT8/INT4 quantized models for faster inference and lower memory usage.

**Implementation notes:**
- ONNX Runtime supports quantized models natively
- Need to add quantized model variants to registry
- Trade-off between speed and accuracy

#### 8. Batched Parallel Processing
Process multiple batches in parallel using threads.

**API design:**
```ruby
embed.embed(documents, batch_size: 32, parallel: 4)
```

**Implementation notes:**
- Ruby's GIL limits true parallelism
- ONNX Runtime already uses threads internally
- May provide marginal benefit for I/O-bound preprocessing

## CLI Enhancements

Future CLI features to consider:

- `fastembed download <model>` - Pre-download models
- `fastembed benchmark` - Run performance benchmarks
- `fastembed info <model>` - Show detailed model information
- Support for reading from files (`-i input.txt`)
- Progress bar for large inputs
- Quiet mode (`-q`) for scripting

## Breaking Changes for v2.0

If we do a major version bump:

1. Consider making `embed()` return an Array instead of Enumerator by default
2. Rename `query_embed`/`passage_embed` to `embed_query`/`embed_passage` for consistency
3. Use keyword arguments consistently throughout

## Contributing

Contributions are welcome! If you'd like to implement any of these features:

1. Open an issue to discuss the approach
2. Follow the existing code style (run `bundle exec rubocop`)
3. Add tests for new functionality
4. Update the README and CHANGELOG
