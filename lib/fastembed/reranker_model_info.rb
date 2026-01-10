# frozen_string_literal: true

module Fastembed
  # Model information for reranker/cross-encoder models
  class RerankerModelInfo
    attr_reader :model_name, :description, :size_in_gb, :model_file,
                :tokenizer_file, :sources

    def initialize(
      model_name:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'onnx/model.onnx',
      tokenizer_file: 'tokenizer.json'
    )
      @model_name = model_name
      @description = description
      @size_in_gb = size_in_gb
      @sources = sources
      @model_file = model_file
      @tokenizer_file = tokenizer_file
    end

    def hf_repo
      sources[:hf]
    end

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
