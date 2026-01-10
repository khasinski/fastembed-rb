# frozen_string_literal: true

module Fastembed
  # Represents a late interaction (ColBERT-style) embedding
  # Contains multiple token-level embeddings instead of a single vector
  class LateInteractionEmbedding
    attr_reader :embeddings, :token_count

    # @param embeddings [Array<Array<Float>>] Token-level embeddings
    def initialize(embeddings)
      @embeddings = embeddings
      @token_count = embeddings.length
    end

    # Get embedding dimension
    # @return [Integer] Dimension of each token embedding
    def dim
      @embeddings.first&.length || 0
    end

    # Compute MaxSim score against another late interaction embedding
    # This is the core ColBERT scoring mechanism
    # @param other [LateInteractionEmbedding] Document embedding to score against
    # @return [Float] MaxSim relevance score
    def max_sim(other)
      return 0.0 if embeddings.empty? || other.embeddings.empty?

      # For each query token, find max similarity with any document token
      embeddings.sum do |query_vec|
        other.embeddings.map do |doc_vec|
          dot_product(query_vec, doc_vec)
        end.max
      end
    end

    def to_a
      @embeddings
    end

    def to_s
      "LateInteractionEmbedding(tokens=#{token_count}, dim=#{dim})"
    end

    def inspect
      to_s
    end

    private

    def dot_product(a, b)
      a.zip(b).sum { |x, y| x * y }
    end
  end

  # Late interaction text embedding using ColBERT-style models
  #
  # Unlike standard embeddings that produce one vector per document,
  # late interaction models produce one vector per token. This enables
  # more fine-grained matching using MaxSim scoring.
  #
  # @example Basic usage
  #   model = Fastembed::LateInteractionTextEmbedding.new
  #   query_emb = model.query_embed("What is ML?").first
  #   doc_emb = model.embed("Machine learning is...").first
  #   score = query_emb.max_sim(doc_emb)
  #
  class LateInteractionTextEmbedding
    include BaseModel

    attr_reader :dim

    # Initialize a late interaction embedding model
    #
    # @param model_name [String] Name of the model to use
    # @param cache_dir [String, nil] Custom cache directory for models
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers
    # @param show_progress [Boolean] Whether to show download progress
    def initialize(
      model_name: DEFAULT_LATE_INTERACTION_MODEL,
      cache_dir: nil,
      threads: nil,
      providers: nil,
      show_progress: true
    )
      initialize_model(
        model_name: model_name,
        cache_dir: cache_dir,
        threads: threads,
        providers: providers,
        show_progress: show_progress
      )

      @dim = @model_info.dim
      setup_model_and_tokenizer
    end

    # Generate late interaction embeddings for documents
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @return [Enumerator] Lazy enumerator yielding LateInteractionEmbedding objects
    def embed(documents, batch_size: 32)
      documents = Validators.validate_documents!(documents)
      return Enumerator.new { |_| } if documents.empty?

      Enumerator.new do |yielder|
        documents.each_slice(batch_size) do |batch|
          embeddings = compute_embeddings(batch)
          embeddings.each { |emb| yielder << emb }
        end
      end
    end

    # Generate late interaction embeddings for queries
    # Queries typically use a special prefix for asymmetric retrieval
    #
    # @param queries [Array<String>, String] Query text(s) to embed
    # @param batch_size [Integer] Number of queries to process at once
    # @return [Enumerator] Lazy enumerator yielding LateInteractionEmbedding objects
    def query_embed(queries, batch_size: 32)
      queries = [queries] if queries.is_a?(String)
      # ColBERT uses [Q] marker for queries
      prefixed = queries.map { |q| "[Q] #{q}" }
      embed(prefixed, batch_size: batch_size)
    end

    # Generate late interaction embeddings for passages/documents
    #
    # @param passages [Array<String>, String] Passage text(s) to embed
    # @param batch_size [Integer] Number of passages to process at once
    # @return [Enumerator] Lazy enumerator yielding LateInteractionEmbedding objects
    def passage_embed(passages, batch_size: 32)
      passages = [passages] if passages.is_a?(String)
      # ColBERT uses [D] marker for documents
      prefixed = passages.map { |p| "[D] #{p}" }
      embed(prefixed, batch_size: batch_size)
    end

    # List all supported late interaction models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_LATE_INTERACTION_MODELS.values.map(&:to_h)
    end

    private

    def resolve_model_info(model_name)
      info = SUPPORTED_LATE_INTERACTION_MODELS[model_name]
      raise Error, "Unknown late interaction model: #{model_name}" unless info

      info
    end

    def compute_embeddings(texts)
      prepared = tokenize_and_prepare(texts)
      outputs = @session.run(nil, prepared[:inputs])
      token_embeddings = extract_token_embeddings(outputs)

      # Create LateInteractionEmbedding for each document
      texts.length.times.map do |i|
        # Filter out padding tokens using attention mask
        valid_embeddings = []
        token_embeddings[i].each_with_index do |emb, j|
          valid_embeddings << normalize_vector(emb) if prepared[:attention_mask][i][j] == 1
        end
        LateInteractionEmbedding.new(valid_embeddings)
      end
    end

    def extract_token_embeddings(outputs)
      if outputs.is_a?(Hash)
        outputs['last_hidden_state'] || outputs['token_embeddings'] || outputs.values.first
      else
        outputs.first
      end
    end

    def normalize_vector(vec)
      norm = Math.sqrt(vec.sum { |x| x * x })
      return vec if norm.zero?

      vec.map { |x| x / norm }
    end
  end
end
