# frozen_string_literal: true

require 'onnxruntime'
require 'tokenizers'

module Fastembed
  # Shared functionality for model classes
  module BaseModel
    attr_reader :model_name, :model_info

    private

    # Common initialization logic for all model types
    def initialize_model(model_name:, cache_dir:, threads:, providers:, show_progress:)
      @model_name = model_name
      @threads = threads
      @providers = providers

      ModelManagement.cache_dir = cache_dir if cache_dir

      @model_info = resolve_model_info(model_name)
      @model_dir = retrieve_model(model_name, show_progress: show_progress)
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
