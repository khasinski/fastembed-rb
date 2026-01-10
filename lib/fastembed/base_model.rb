# frozen_string_literal: true

require 'onnxruntime'
require 'tokenizers'

module Fastembed
  # Shared functionality for model classes
  #
  # This module provides common initialization and utility methods used by
  # all model types (TextEmbedding, TextCrossEncoder, TextSparseEmbedding, etc.).
  # It handles model downloading, ONNX session creation, and tokenizer loading.
  #
  # @abstract Include in model classes and call {#initialize_model}
  #
  module BaseModel
    # @!attribute [r] model_name
    #   @return [String] Name of the loaded model
    # @!attribute [r] model_info
    #   @return [BaseModelInfo] Model metadata and configuration
    # @!attribute [r] quantization
    #   @return [Symbol] Current quantization type
    attr_reader :model_name, :model_info, :quantization

    private

    # Common initialization logic for all model types
    # @param model_name [String] Name of the model
    # @param cache_dir [String, nil] Custom cache directory
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers
    # @param show_progress [Boolean] Whether to show download progress
    # @param quantization [Symbol] Quantization type (:fp32, :fp16, :int8, :uint8, :q4)
    def initialize_model(model_name:, cache_dir:, threads:, providers:, show_progress:, quantization: nil)
      @model_name = model_name
      @threads = threads
      @providers = providers
      @quantization = quantization || Quantization::DEFAULT

      validate_quantization!

      ModelManagement.cache_dir = cache_dir if cache_dir

      @model_info = resolve_model_info(model_name)
      @model_dir = retrieve_model(model_name, show_progress: show_progress)
    end

    # Validate that the quantization type is supported
    # @raise [ArgumentError] If quantization type is invalid
    def validate_quantization!
      return if Quantization.valid?(@quantization)

      valid_types = Quantization::TYPES.keys.join(', ')
      raise ArgumentError, "Invalid quantization type: #{@quantization}. Valid types: #{valid_types}"
    end

    # Get the model file path, accounting for quantization
    # @return [String] Path to quantized model file (or base if fp32)
    def quantized_model_file
      Quantization.model_file(@model_info.model_file, @quantization)
    end

    # Override in subclasses to resolve from appropriate registry
    #
    # @param _model_name [String] Name of the model
    # @return [BaseModelInfo] Model information object
    # @raise [NotImplementedError] If not overridden in subclass
    # @abstract
    def resolve_model_info(_model_name)
      raise NotImplementedError, 'Subclasses must implement resolve_model_info'
    end

    # Download or retrieve cached model
    #
    # @param model_name [String] Name of the model
    # @param show_progress [Boolean] Whether to show download progress
    # @return [String] Path to model directory
    def retrieve_model(model_name, show_progress:)
      ModelManagement.retrieve_model(
        model_name,
        model_info: @model_info,
        show_progress: show_progress
      )
    end

    # Build ONNX session options hash
    # @return [Hash] Options for OnnxRuntime::InferenceSession
    def build_session_options
      options = {}
      options[:inter_op_num_threads] = @threads if @threads
      options[:intra_op_num_threads] = @threads if @threads
      options
    end

    # Load an ONNX inference session
    #
    # @param model_path [String] Path to ONNX model file
    # @param providers [Array<String>, nil] Execution providers
    # @return [OnnxRuntime::InferenceSession] The loaded session
    # @raise [Error] If model file not found
    def load_onnx_session(model_path, providers: nil)
      raise Error, "Model file not found: #{model_path}" unless File.exist?(model_path)

      OnnxRuntime::InferenceSession.new(
        model_path,
        **build_session_options,
        providers: providers || ['CPUExecutionProvider']
      )
    end

    # Load a HuggingFace tokenizer from file
    #
    # @param tokenizer_path [String] Path to tokenizer.json
    # @param max_length [Integer] Maximum sequence length for truncation
    # @return [Tokenizers::Tokenizer] Configured tokenizer
    # @raise [Error] If tokenizer file not found
    def load_tokenizer_from_file(tokenizer_path, max_length:)
      raise Error, "Tokenizer not found: #{tokenizer_path}" unless File.exist?(tokenizer_path)

      tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_path)
      tokenizer.enable_padding(pad_id: 0, pad_token: '[PAD]')
      tokenizer.enable_truncation(max_length)
      tokenizer
    end
  end
end
