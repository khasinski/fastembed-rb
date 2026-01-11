# frozen_string_literal: true

module Fastembed
  # Model information for late interaction models (ColBERT, etc.)
  #
  # Late interaction models produce token-level embeddings instead of a single
  # document vector, enabling fine-grained matching via MaxSim scoring.
  #
  # @example Access late interaction model info
  #   info = Fastembed::SUPPORTED_LATE_INTERACTION_MODELS['colbert-ir/colbertv2.0']
  #   info.dim  # => 128
  #
  class LateInteractionModelInfo
    include BaseModelInfo

    # @!attribute [r] dim
    #   @return [Integer] Output embedding dimension per token
    attr_reader :dim

    # Create a new LateInteractionModelInfo instance
    #
    # @param model_name [String] Full model identifier
    # @param dim [Integer] Output embedding dimension per token
    # @param description [String] Human-readable description
    # @param size_in_gb [Float] Model size in GB
    # @param sources [Hash] Source repositories
    # @param model_file [String] Path to ONNX model file
    # @param tokenizer_file [String] Path to tokenizer file
    # @param max_length [Integer] Maximum sequence length
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

    # Convert to hash representation
    # @return [Hash] Model info as a hash
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
      model_file: 'model.onnx',
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
