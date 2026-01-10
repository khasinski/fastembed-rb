# frozen_string_literal: true

module Fastembed
  # Model information for late interaction models (ColBERT, etc.)
  class LateInteractionModelInfo
    include BaseModelInfo

    attr_reader :dim

    def initialize(
      model_name:,
      dim:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'onnx/model.onnx',
      tokenizer_file: 'tokenizer.json',
      max_length: 512
    )
      initialize_base(
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file,
        max_length: max_length
      )
      @dim = dim
    end

    def to_h
      {
        model_name: model_name,
        dim: dim,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file,
        max_length: max_length
      }
    end
  end

  # Registry of supported late interaction models
  SUPPORTED_LATE_INTERACTION_MODELS = {
    'colbert-ir/colbertv2.0' => LateInteractionModelInfo.new(
      model_name: 'colbert-ir/colbertv2.0',
      dim: 128,
      description: 'ColBERTv2 for late interaction retrieval',
      size_in_gb: 0.44,
      sources: { hf: 'colbert-ir/colbertv2.0' },
      max_length: 512
    ),
    'jinaai/jina-colbert-v1-en' => LateInteractionModelInfo.new(
      model_name: 'jinaai/jina-colbert-v1-en',
      dim: 128,
      description: 'Jina ColBERT v1 for English with 8192 context',
      size_in_gb: 0.55,
      sources: { hf: 'jinaai/jina-colbert-v1-en' },
      max_length: 8192
    )
  }.freeze

  DEFAULT_LATE_INTERACTION_MODEL = 'colbert-ir/colbertv2.0'
end
