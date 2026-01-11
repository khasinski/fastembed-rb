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
    # @param quantization [Symbol] Quantization type (:fp32, :fp16, :int8, :uint8, :q4)
    # @param local_model_dir [String, nil] Load model from local directory instead of downloading
    # @param model_file [String, nil] Override model file name (e.g., "model.onnx")
    # @param tokenizer_file [String, nil] Override tokenizer file name (e.g., "tokenizer.json")
    def initialize(
      model_name: DEFAULT_LATE_INTERACTION_MODEL,
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
      setup_model_and_tokenizer(model_file_override: model_file || quantized_model_file)
    end

    # Generate late interaction embeddings for documents
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @yield [Progress] Optional progress callback called after each batch
    # @return [Enumerator] Lazy enumerator yielding LateInteractionEmbedding objects
    def embed(documents, batch_size: 32, &progress_callback)
      documents = Validators.validate_documents!(documents)
      return Enumerator.new { |_| } if documents.empty?

      total_batches = (documents.length.to_f / batch_size).ceil

      Enumerator.new do |yielder|
        documents.each_slice(batch_size).with_index(1) do |batch, batch_num|
          embeddings = compute_embeddings(batch)
          embeddings.each { |emb| yielder << emb }

          if progress_callback
            progress = Progress.new(current: batch_num, total: total_batches, batch_size: batch_size)
            progress_callback.call(progress)
          end
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

    # Generate embeddings asynchronously
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @return [Async::Future] Future that resolves to array of LateInteractionEmbedding objects
    def embed_async(documents, batch_size: 32)
      Async::Future.new { embed(documents, batch_size: batch_size).to_a }
    end

    # Generate query embeddings asynchronously
    #
    # @param queries [Array<String>, String] Query text(s) to embed
    # @param batch_size [Integer] Number of queries to process at once
    # @return [Async::Future] Future that resolves to array of LateInteractionEmbedding objects
    def query_embed_async(queries, batch_size: 32)
      Async::Future.new { query_embed(queries, batch_size: batch_size).to_a }
    end

    # Generate passage embeddings asynchronously
    #
    # @param passages [Array<String>, String] Passage text(s) to embed
    # @param batch_size [Integer] Number of passages to process at once
    # @return [Async::Future] Future that resolves to array of LateInteractionEmbedding objects
    def passage_embed_async(passages, batch_size: 32)
      Async::Future.new { passage_embed(passages, batch_size: batch_size).to_a }
    end

    # List all supported late interaction models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_LATE_INTERACTION_MODELS.values.map(&:to_h)
    end

    private

    def resolve_model_info(model_name)
      # Check built-in registry first
      info = SUPPORTED_LATE_INTERACTION_MODELS[model_name]
      return info if info

      # Check custom registry
      info = CustomModelRegistry.late_interaction_models[model_name]
      return info if info

      raise Error, "Unknown late interaction model: #{model_name}"
    end

    def create_local_model_info(model_name:, model_file:, tokenizer_file:)
      LateInteractionModelInfo.new(
        model_name: model_name,
        description: 'Local late interaction model',
        size_in_gb: 0,
        sources: {},
        model_file: model_file || 'model.onnx',
        tokenizer_file: tokenizer_file || 'tokenizer.json',
        dim: 128 # Default ColBERT dimension
      )
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
