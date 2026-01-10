# frozen_string_literal: true

require 'onnxruntime'
require 'tokenizers'

module Fastembed
  # Shared functionality for model classes
  module BaseModel
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

    def validate_quantization!
      return if Quantization.valid?(@quantization)

      valid_types = Quantization::TYPES.keys.join(', ')
      raise ArgumentError, "Invalid quantization type: #{@quantization}. Valid types: #{valid_types}"
    end

    # Get the model file path, accounting for quantization
    def quantized_model_file
      Quantization.model_file(@model_info.model_file, @quantization)
    end

    # Override in subclasses to resolve from appropriate registry
    def resolve_model_info(_model_name)
      raise NotImplementedError, 'Subclasses must implement resolve_model_info'
    end

    def retrieve_model(model_name, show_progress:)
      ModelManagement.retrieve_model(
        model_name,
        model_info: @model_info,
        show_progress: show_progress
      )
    end

    def build_session_options
      options = {}
      options[:inter_op_num_threads] = @threads if @threads
      options[:intra_op_num_threads] = @threads if @threads
      options
    end

    def load_onnx_session(model_path, providers: nil)
      raise Error, "Model file not found: #{model_path}" unless File.exist?(model_path)

      OnnxRuntime::InferenceSession.new(
        model_path,
        **build_session_options,
        providers: providers || ['CPUExecutionProvider']
      )
    end

    def load_tokenizer_from_file(tokenizer_path, max_length:)
      raise Error, "Tokenizer not found: #{tokenizer_path}" unless File.exist?(tokenizer_path)

      tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_path)
      tokenizer.enable_padding(pad_id: 0, pad_token: '[PAD]')
      tokenizer.enable_truncation(max_length)
      tokenizer
    end
  end
end
