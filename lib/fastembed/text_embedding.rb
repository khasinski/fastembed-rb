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
  class TextEmbedding
    attr_reader :model_name, :model_info, :dim

    # Initialize a text embedding model
    #
    # @param model_name [String] Name of the model to use (default: "BAAI/bge-small-en-v1.5")
    # @param cache_dir [String, nil] Custom cache directory for models
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers (e.g., ["CoreMLExecutionProvider"])
    # @param show_progress [Boolean] Whether to show download progress
    def initialize(
      model_name: DEFAULT_MODEL,
      cache_dir: nil,
      threads: nil,
      providers: nil,
      show_progress: true
    )
      @model_name = model_name
      @threads = threads
      @providers = providers
      @show_progress = show_progress

      # Set custom cache directory if provided
      ModelManagement.cache_dir = cache_dir if cache_dir

      # Resolve model info
      @model_info = ModelManagement.resolve_model_info(model_name)
      @dim = @model_info.dim

      # Download and load model
      @model_dir = ModelManagement.retrieve_model(model_name, show_progress: show_progress)
      @model = OnnxEmbeddingModel.new(@model_info, @model_dir, threads: threads, providers: providers)
    end

    # Generate embeddings for documents
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @return [Enumerator] Lazy enumerator yielding embedding vectors
    #
    # @example
    #   vectors = embedding.embed(["Hello", "World"]).to_a
    #   # => [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    def embed(documents, batch_size: 256)
      documents = [documents] if documents.is_a?(String)

      Enumerator.new do |yielder|
        documents.each_slice(batch_size) do |batch|
          embeddings = @model.embed(batch)
          embeddings.each { |embedding| yielder << embedding }
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

    # List all supported models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_MODELS.values.map(&:to_h)
    end

    # Get information about a specific model
    #
    # @param model_name [String] Name of the model
    # @return [Hash, nil] Model information or nil if not found
    def self.get_model_info(model_name)
      SUPPORTED_MODELS[model_name]&.to_h
    end
  end
end
