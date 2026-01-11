# frozen_string_literal: true

module Fastembed
  # Model information for sparse embedding models (SPLADE, etc.)
  #
  # Sparse models like SPLADE produce vectors where most dimensions are zero,
  # with non-zero values corresponding to vocabulary tokens. These are useful
  # for hybrid search combining with dense embeddings.
  #
  # @example Access sparse model info
  #   info = Fastembed::SUPPORTED_SPARSE_MODELS['prithivida/Splade_PP_en_v1']
  #   info.model_name  # => "prithivida/Splade_PP_en_v1"
  #
  class SparseModelInfo
    include BaseModelInfo

    # Create a new SparseModelInfo instance
    #
    # @param model_name [String] Full model identifier
    # @param description [String] Human-readable description
    # @param size_in_gb [Float] Model size in GB
    # @param sources [Hash] Source repositories
    # @param model_file [String] Path to ONNX model file
    # @param tokenizer_file [String] Path to tokenizer file
    # @param max_length [Integer] Maximum sequence length
    def initialize(
      model_name:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'onnx/model.onnx',
      tokenizer_file: 'tokenizer.json',
      max_length: BaseModelInfo::DEFAULT_MAX_LENGTH
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
        tokenizer_file: tokenizer_file,
        max_length: max_length
      }
    end
  end

  # Registry of supported sparse embedding models
  SUPPORTED_SPARSE_MODELS = {
    'prithivida/Splade_PP_en_v1' => SparseModelInfo.new(
      model_name: 'prithivida/Splade_PP_en_v1',
      description: 'SPLADE++ model for sparse text retrieval',
      size_in_gb: 0.53,
      sources: { hf: 'prithivida/Splade_PP_en_v1' }
      # Uses default model_file: 'onnx/model.onnx'
    ),
    'prithivida/Splade_PP_en_v2' => SparseModelInfo.new(
      model_name: 'prithivida/Splade_PP_en_v2',
      description: 'SPLADE++ v2 with improved performance',
      size_in_gb: 0.53,
      sources: { hf: 'prithivida/Splade_PP_en_v2' }
      # Uses default model_file: 'onnx/model.onnx'
    )
  }.freeze

  DEFAULT_SPARSE_MODEL = 'prithivida/Splade_PP_en_v1'
end
