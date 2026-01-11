# frozen_string_literal: true

module Fastembed
  # Main class for generating text embeddings
  #
  # @example Basic usage
  #   embedding = Fastembed::TextEmbedding.new
  #   vectors = embedding.embed(["Hello world", "Another text"]).to_a
  #
  # @example Custom model
  #   embedding = Fastembed::TextEmbedding.new(model_name: "BAAI/bge-base-en-v1.5")
  #
  # @example Lazy iteration for large datasets
  #   embedding.embed(documents).each do |vector|
  #     # Process each vector
  #   end
  #
  # @example Load from local directory
  #   embedding = Fastembed::TextEmbedding.new(
  #     local_model_dir: "/path/to/model",
  #     model_file: "model.onnx",
  #     tokenizer_file: "tokenizer.json"
  #   )
  #
  class TextEmbedding
    include BaseModel

    attr_reader :dim

    # Initialize a text embedding model
    #
    # @param model_name [String] Name of the model to use (default: "BAAI/bge-small-en-v1.5")
    # @param cache_dir [String, nil] Custom cache directory for models
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers (e.g., ["CoreMLExecutionProvider"])
    # @param show_progress [Boolean] Whether to show download progress
    # @param quantization [Symbol] Quantization type (:fp32, :fp16, :int8, :uint8, :q4)
    # @param local_model_dir [String, nil] Load model from local directory instead of downloading
    # @param model_file [String, nil] Override model file name (e.g., "model.onnx")
    # @param tokenizer_file [String, nil] Override tokenizer file name (e.g., "tokenizer.json")
    def initialize(
      model_name: DEFAULT_MODEL,
      cache_dir: nil,
      threads: nil,
      providers: nil,
      show_progress: true,
      quantization: nil,
      local_model_dir: nil,
      model_file: nil,
      tokenizer_file: nil
    )
      if local_model_dir
        initialize_from_local(
          local_model_dir: local_model_dir,
          model_name: model_name,
          threads: threads,
          providers: providers,
          quantization: quantization,
          model_file: model_file,
          tokenizer_file: tokenizer_file
        )
      else
        initialize_model(
          model_name: model_name,
          cache_dir: cache_dir,
          threads: threads,
          providers: providers,
          show_progress: show_progress,
          quantization: quantization
        )
      end

      @dim = @model_info.dim
      @model = OnnxEmbeddingModel.new(
        @model_info,
        @model_dir,
        threads: threads,
        providers: providers,
        model_file_override: model_file || quantized_model_file
      )
    end

    # Generate embeddings for documents
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @yield [Progress] Optional progress callback called after each batch
    # @return [Enumerator] Lazy enumerator yielding embedding vectors
    # @raise [ArgumentError] If documents is nil or contains nil values
    #
    # @example Basic usage
    #   vectors = embedding.embed(["Hello", "World"]).to_a
    #   # => [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    #
    # @example With progress callback
    #   embedding.embed(documents, batch_size: 64) do |progress|
    #     puts "#{progress.percent}% complete"
    #   end.to_a
    #
    def embed(documents, batch_size: 256, &progress_callback)
      documents = Validators.validate_documents!(documents)
      return Enumerator.new { |_| } if documents.empty?

      total_batches = (documents.length.to_f / batch_size).ceil

      Enumerator.new do |yielder|
        documents.each_slice(batch_size).with_index(1) do |batch, batch_num|
          embeddings = @model.embed(batch)
          embeddings.each { |embedding| yielder << embedding }

          if progress_callback
            progress = Progress.new(current: batch_num, total: total_batches, batch_size: batch_size)
            progress_callback.call(progress)
          end
        end
      end
    end

    # Generate embeddings for query texts (with "query: " prefix for retrieval models)
    #
    # @param queries [Array<String>, String] Query text(s) to embed
    # @param batch_size [Integer] Number of queries to process at once
    # @return [Enumerator] Lazy enumerator yielding embedding vectors
    def query_embed(queries, batch_size: 256)
      queries = [queries] if queries.is_a?(String)
      prefixed = queries.map { |q| "query: #{q}" }
      embed(prefixed, batch_size: batch_size)
    end

    # Generate embeddings for passage texts (with "passage: " prefix for retrieval models)
    #
    # @param passages [Array<String>, String] Passage text(s) to embed
    # @param batch_size [Integer] Number of passages to process at once
    # @return [Enumerator] Lazy enumerator yielding embedding vectors
    def passage_embed(passages, batch_size: 256)
      passages = [passages] if passages.is_a?(String)
      prefixed = passages.map { |p| "passage: #{p}" }
      embed(prefixed, batch_size: batch_size)
    end

    # Generate embeddings asynchronously in a background thread
    #
    # Returns immediately with a Future object. The embedding computation
    # runs in a background thread. Call `value` on the Future to get results.
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @return [Async::Future] Future that resolves to array of embedding vectors
    #
    # @example Basic async usage
    #   future = embedding.embed_async(documents)
    #   # ... do other work ...
    #   vectors = future.value  # blocks until complete
    #
    # @example Parallel embedding of multiple batches
    #   futures = documents.each_slice(1000).map do |batch|
    #     embedding.embed_async(batch)
    #   end
    #   all_vectors = futures.flat_map(&:value)
    #
    def embed_async(documents, batch_size: 256)
      Async::Future.new do
        embed(documents, batch_size: batch_size).to_a
      end
    end

    # Generate query embeddings asynchronously
    #
    # @param queries [Array<String>, String] Query text(s) to embed
    # @param batch_size [Integer] Number of queries to process at once
    # @return [Async::Future] Future that resolves to array of embedding vectors
    def query_embed_async(queries, batch_size: 256)
      Async::Future.new do
        query_embed(queries, batch_size: batch_size).to_a
      end
    end

    # Generate passage embeddings asynchronously
    #
    # @param passages [Array<String>, String] Passage text(s) to embed
    # @param batch_size [Integer] Number of passages to process at once
    # @return [Async::Future] Future that resolves to array of embedding vectors
    def passage_embed_async(passages, batch_size: 256)
      Async::Future.new do
        passage_embed(passages, batch_size: batch_size).to_a
      end
    end

    # List all supported models (built-in and custom)
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      all_models = SUPPORTED_MODELS.merge(CustomModelRegistry.embedding_models)
      all_models.values.map(&:to_h)
    end

    # Get information about a specific model
    #
    # @param model_name [String] Name of the model
    # @return [Hash, nil] Model information or nil if not found
    def self.get_model_info(model_name)
      info = SUPPORTED_MODELS[model_name] || CustomModelRegistry.embedding_models[model_name]
      info&.to_h
    end

    private

    def resolve_model_info(model_name)
      ModelManagement.resolve_model_info(model_name)
    end

    def initialize_from_local(local_model_dir:, model_name:, threads:, providers:, quantization:, model_file:, tokenizer_file:)
      raise ArgumentError, "Local model directory not found: #{local_model_dir}" unless Dir.exist?(local_model_dir)

      @model_name = model_name
      @threads = threads
      @providers = providers
      @quantization = quantization || Quantization::DEFAULT
      @model_dir = local_model_dir

      validate_quantization!

      # Try to get model info from registry (built-in or custom), or create a minimal one
      @model_info = SUPPORTED_MODELS[model_name] ||
                    CustomModelRegistry.embedding_models[model_name] ||
                    create_local_model_info(
        model_name: model_name,
        model_file: model_file,
        tokenizer_file: tokenizer_file
      )
    end

    def create_local_model_info(model_name:, model_file:, tokenizer_file:)
      # Detect dimension from model output shape if possible
      # For now, use a placeholder that will be updated after model load
      ModelInfo.new(
        model_name: model_name,
        dim: detect_model_dimension(model_file) || 384,
        description: 'Local model',
        size_in_gb: 0,
        sources: {},
        model_file: model_file || 'model.onnx',
        tokenizer_file: tokenizer_file || 'tokenizer.json'
      )
    end

    def detect_model_dimension(model_file)
      # Try to detect dimension from ONNX model metadata
      model_path = File.join(@model_dir, model_file || 'model.onnx')
      return nil unless File.exist?(model_path)

      begin
        session = OnnxRuntime::InferenceSession.new(model_path)
        # Look for output shape - usually [batch, seq_len, hidden_size] or [batch, hidden_size]
        output = session.outputs.first
        return nil unless output && output[:shape]

        shape = output[:shape]
        # Last dimension is usually the embedding dimension
        dim = shape.last
        dim.is_a?(Integer) && dim > 0 ? dim : nil
      rescue StandardError
        nil
      end
    end
  end
end
