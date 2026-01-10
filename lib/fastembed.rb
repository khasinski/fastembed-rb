# frozen_string_literal: true

require_relative 'fastembed/version'

# Fastembed - Fast, lightweight text embeddings for Ruby
#
# A Ruby port of FastEmbed providing text embeddings using ONNX Runtime.
# Supports dense embeddings, sparse embeddings (SPLADE), late interaction (ColBERT),
# and cross-encoder reranking.
#
# @example Basic text embedding
#   embedding = Fastembed::TextEmbedding.new
#   vectors = embedding.embed(["Hello world", "Ruby is great"]).to_a
#
# @example Reranking documents
#   reranker = Fastembed::TextCrossEncoder.new
#   scores = reranker.rerank(query: "What is ML?", documents: docs)
#
# @example Sparse embeddings
#   sparse = Fastembed::TextSparseEmbedding.new
#   embeddings = sparse.embed(["Hello"]).to_a
#
# @see https://github.com/khasinski/fastembed-rb
#
module Fastembed
  # Base error class for all Fastembed errors
  class Error < StandardError; end

  # Raised when model download fails
  class DownloadError < Error; end
end

require_relative 'fastembed/base_model_info'
require_relative 'fastembed/model_info'
require_relative 'fastembed/reranker_model_info'
require_relative 'fastembed/sparse_model_info'
require_relative 'fastembed/late_interaction_model_info'
require_relative 'fastembed/model_management'
require_relative 'fastembed/pooling'
require_relative 'fastembed/quantization'
require_relative 'fastembed/progress'
require_relative 'fastembed/base_model'
require_relative 'fastembed/onnx_embedding_model'
require_relative 'fastembed/text_embedding'
require_relative 'fastembed/text_cross_encoder'
require_relative 'fastembed/sparse_embedding'
require_relative 'fastembed/late_interaction_embedding'
