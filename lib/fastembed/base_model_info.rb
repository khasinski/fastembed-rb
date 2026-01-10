# frozen_string_literal: true

module Fastembed
  # Shared functionality for model information classes
  #
  # This module provides common attributes and methods for all model info types
  # (embedding, reranking, sparse, late interaction). It handles metadata like
  # model name, file paths, and HuggingFace source information.
  #
  # @abstract Include in model info classes and call {#initialize_base}
  #
  module BaseModelInfo
    # Default maximum sequence length for tokenization
    DEFAULT_MAX_LENGTH = 512

    # @!attribute [r] model_name
    #   @return [String] Full model identifier (e.g., "BAAI/bge-small-en-v1.5")
    # @!attribute [r] description
    #   @return [String] Human-readable model description
    # @!attribute [r] size_in_gb
    #   @return [Float] Approximate model size in gigabytes
    # @!attribute [r] model_file
    #   @return [String] Relative path to ONNX model file
    # @!attribute [r] tokenizer_file
    #   @return [String] Relative path to tokenizer JSON file
    # @!attribute [r] sources
    #   @return [Hash] Source repositories (e.g., {hf: "Xenova/model-name"})
    # @!attribute [r] max_length
    #   @return [Integer] Maximum token sequence length
    attr_reader :model_name, :description, :size_in_gb, :model_file,
                :tokenizer_file, :sources, :max_length

    # Returns the HuggingFace repository ID
    # @return [String] The HF repo ID for downloading
    def hf_repo
      sources[:hf]
    end

    private

    # Initialize common model info attributes
    #
    # @param model_name [String] Full model identifier
    # @param description [String] Human-readable description
    # @param size_in_gb [Float] Model size in GB
    # @param sources [Hash] Source repositories
    # @param model_file [String] Path to ONNX model file
    # @param tokenizer_file [String] Path to tokenizer file
    # @param max_length [Integer] Maximum sequence length
    def initialize_base(model_name:, description:, size_in_gb:, sources:, model_file:, tokenizer_file:,
                        max_length: DEFAULT_MAX_LENGTH)
      @model_name = model_name
      @description = description
      @size_in_gb = size_in_gb
      @sources = sources
      @model_file = model_file
      @tokenizer_file = tokenizer_file
      @max_length = max_length
    end
  end
end
