# frozen_string_literal: true

require 'onnxruntime'
require 'tokenizers'

module Fastembed
  # ONNX-based embedding model wrapper
  class OnnxEmbeddingModel
    attr_reader :model_info, :model_dir

    # @param model_info [ModelInfo] Model metadata
    # @param model_dir [String] Directory containing model files
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers
    # @param model_file_override [String, nil] Override model file path (for quantized models)
    def initialize(model_info, model_dir, threads: nil, providers: nil, model_file_override: nil)
      @model_info = model_info
      @model_dir = model_dir
      @threads = threads
      @providers = providers
      @model_file = model_file_override || model_info.model_file

      load_model
      load_tokenizer
    end

    # Embed a batch of texts
    def embed(texts)
      # Tokenize
      encoded = tokenize(texts)

      # Run inference
      outputs = run_inference(encoded)

      # Apply pooling and normalization
      Pooling.apply(
        model_info.pooling,
        outputs,
        encoded[:attention_mask],
        should_normalize: model_info.normalize
      )
    end

    private

    def load_model
      model_path = File.join(model_dir, @model_file)
      raise Error, "Model file not found: #{model_path}" unless File.exist?(model_path)

      session_options = {}
      session_options[:inter_op_num_threads] = @threads if @threads
      session_options[:intra_op_num_threads] = @threads if @threads
      session_options[:providers] = @providers if @providers

      @session = OnnxRuntime::InferenceSession.new(model_path, **session_options)
    end

    def load_tokenizer
      tokenizer_path = File.join(model_dir, model_info.tokenizer_file)
      raise Error, "Tokenizer file not found: #{tokenizer_path}" unless File.exist?(tokenizer_path)

      @tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_path)

      # Configure tokenizer for batch encoding
      @tokenizer.enable_padding(pad_id: 0, pad_token: '[PAD]')
      @tokenizer.enable_truncation(model_info.max_length)
    end

    def tokenize(texts)
      texts = [texts] if texts.is_a?(String)

      # Batch encode
      encodings = @tokenizer.encode_batch(texts)

      # Convert to format expected by ONNX model
      {
        input_ids: encodings.map(&:ids),
        attention_mask: encodings.map(&:attention_mask),
        token_type_ids: encodings.map { |e| e.type_ids || Array.new(e.ids.length, 0) }
      }
    end

    def run_inference(encoded)
      # Prepare inputs
      inputs = {
        'input_ids' => encoded[:input_ids],
        'attention_mask' => encoded[:attention_mask]
      }

      # Some models require token_type_ids
      inputs['token_type_ids'] = encoded[:token_type_ids] if input_names.include?('token_type_ids')

      # Run model
      outputs = @session.run(nil, inputs)

      # Get the last hidden state (usually first output)
      # Output shape: [batch_size, seq_len, hidden_size]
      extract_embeddings(outputs)
    end

    def input_names
      @input_names ||= @session.inputs.map { |i| i[:name] }
    end

    def output_names
      @output_names ||= @session.outputs.map { |o| o[:name] }
    end

    def extract_embeddings(outputs)
      # ONNX models typically output as a hash or the first output
      # The key is usually "last_hidden_state" or similar
      if outputs.is_a?(Hash)
        # Try common output names
        %w[last_hidden_state token_embeddings].each do |key|
          return outputs[key] if outputs.key?(key)
        end
        # Fall back to first output
        outputs.values.first
      else
        outputs.first
      end
    end
  end
end
