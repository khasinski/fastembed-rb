# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-01-08

### Added

- Initial release
- `TextEmbedding` class for generating text embeddings
- Automatic model downloading and caching from HuggingFace
- Support for multiple embedding models:
  - BAAI/bge-small-en-v1.5 (default)
  - BAAI/bge-base-en-v1.5
  - BAAI/bge-large-en-v1.5
  - sentence-transformers/all-MiniLM-L6-v2
  - intfloat/multilingual-e5-small
  - intfloat/multilingual-e5-base
- Lazy evaluation with `Enumerator` for memory efficiency
- Query and passage embedding methods for retrieval tasks
- Configurable batch size, threading, and execution providers
- Mean pooling and L2 normalization
- CoreML execution provider support (experimental)

[Unreleased]: https://github.com/khasinski/fastembed-rb/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/khasinski/fastembed-rb/releases/tag/v1.0.0
