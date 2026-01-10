# frozen_string_literal: true

module Fastembed
  # Model information for reranker/cross-encoder models
  #
  # Cross-encoders process query-document pairs together rather than
  # encoding them separately, enabling more accurate relevance scoring.
  #
  # @example Access reranker info
  #   info = Fastembed::SUPPORTED_RERANKER_MODELS['BAAI/bge-reranker-base']
  #   info.model_name  # => "BAAI/bge-reranker-base"
  #
  class RerankerModelInfo
    include BaseModelInfo

    # Create a new RerankerModelInfo instance
    #
    # @param model_name [String] Full model identifier
    # @param description [String] Human-readable description
    # @param size_in_gb [Float] Model size in GB
    # @param sources [Hash] Source repositories
    # @param model_file [String] Path to ONNX model file
    # @param tokenizer_file [String] Path to tokenizer file
    def initialize(
      model_name:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'onnx/model.onnx',
      tokenizer_file: 'tokenizer.json'
    )
      initialize_base(
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file
      )
    end

    # Convert to hash representation
    # @return [Hash] Model info as a hash
    def to_h
      {
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file
      }
    end
  end

  # Registry of supported reranker models
  SUPPORTED_RERANKER_MODELS = {
    'BAAI/bge-reranker-base' => RerankerModelInfo.new(
      model_name: 'BAAI/bge-reranker-base',
      description: 'BGE reranker base model for relevance scoring',
      size_in_gb: 1.11,
      sources: { hf: 'Xenova/bge-reranker-base' }
    ),
    'BAAI/bge-reranker-large' => RerankerModelInfo.new(
      model_name: 'BAAI/bge-reranker-large',
      description: 'BGE reranker large model for higher accuracy',
      size_in_gb: 2.24,
      sources: { hf: 'Xenova/bge-reranker-large' }
    ),
    'cross-encoder/ms-marco-MiniLM-L-6-v2' => RerankerModelInfo.new(
      model_name: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
      description: 'Lightweight MS MARCO cross-encoder',
      size_in_gb: 0.08,
      sources: { hf: 'Xenova/ms-marco-MiniLM-L-6-v2' }
    ),
    'cross-encoder/ms-marco-MiniLM-L-12-v2' => RerankerModelInfo.new(
      model_name: 'cross-encoder/ms-marco-MiniLM-L-12-v2',
      description: 'MS MARCO cross-encoder with better accuracy',
      size_in_gb: 0.12,
      sources: { hf: 'Xenova/ms-marco-MiniLM-L-12-v2' }
    ),
    'jinaai/jina-reranker-v1-turbo-en' => RerankerModelInfo.new(
      model_name: 'jinaai/jina-reranker-v1-turbo-en',
      description: 'Fast Jina reranker for English',
      size_in_gb: 0.13,
      sources: { hf: 'jinaai/jina-reranker-v1-turbo-en' }
    )
  }.freeze

  DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
end
