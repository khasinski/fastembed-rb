# frozen_string_literal: true

module Fastembed
  # Represents a sparse embedding vector
  # Contains indices and their corresponding values (non-zero only)
  class SparseEmbedding
    attr_reader :indices, :values

    def initialize(indices:, values:)
      @indices = indices
      @values = values
    end

    # Convert to hash format {index => value}
    def to_h
      indices.zip(values).to_h
    end

    # Number of non-zero elements
    def nnz
      indices.length
    end

    def to_s
      "SparseEmbedding(nnz=#{nnz})"
    end

    def inspect
      to_s
    end
  end

  # Sparse text embedding using SPLADE models
  #
  # SPLADE (Sparse Lexical and Expansion) models produce sparse vectors
  # where each dimension corresponds to a vocabulary token.
  # These are useful for hybrid search combining with dense embeddings.
  #
  # @example Basic usage
  #   sparse = Fastembed::TextSparseEmbedding.new
  #   embeddings = sparse.embed(["Hello world"]).to_a
  #   embeddings.first.indices  # => [101, 7592, 2088, ...]
  #   embeddings.first.values   # => [0.45, 1.23, 0.89, ...]
  #
  class TextSparseEmbedding
    include BaseModel

    # Initialize a sparse embedding model
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
      model_name: DEFAULT_SPARSE_MODEL,
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

      setup_model_and_tokenizer(model_file_override: model_file || quantized_model_file)
    end

    # Generate sparse embeddings for documents
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @yield [Progress] Optional progress callback called after each batch
    # @return [Enumerator] Lazy enumerator yielding SparseEmbedding objects
    def embed(documents, batch_size: 32, &progress_callback)
      documents = Validators.validate_documents!(documents)
      return Enumerator.new { |_| } if documents.empty?

      total_batches = (documents.length.to_f / batch_size).ceil

      Enumerator.new do |yielder|
        documents.each_slice(batch_size).with_index(1) do |batch, batch_num|
          sparse_embeddings = compute_sparse_embeddings(batch)
          sparse_embeddings.each { |emb| yielder << emb }

          if progress_callback
            progress = Progress.new(current: batch_num, total: total_batches, batch_size: batch_size)
            progress_callback.call(progress)
          end
        end
      end
    end

    # Generate sparse embeddings for queries (with "query: " prefix)
    #
    # @param queries [Array<String>, String] Query text(s) to embed
    # @param batch_size [Integer] Number of queries to process at once
    # @return [Enumerator] Lazy enumerator yielding SparseEmbedding objects
    def query_embed(queries, batch_size: 32)
      queries = [queries] if queries.is_a?(String)
      prefixed = queries.map { |q| "query: #{q}" }
      embed(prefixed, batch_size: batch_size)
    end

    # Generate sparse embeddings for passages
    #
    # @param passages [Array<String>, String] Passage text(s) to embed
    # @param batch_size [Integer] Number of passages to process at once
    # @return [Enumerator] Lazy enumerator yielding SparseEmbedding objects
    def passage_embed(passages, batch_size: 32)
      passages = [passages] if passages.is_a?(String)
      embed(passages, batch_size: batch_size)
    end

    # Generate sparse embeddings asynchronously
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @return [Async::Future] Future that resolves to array of SparseEmbedding objects
    def embed_async(documents, batch_size: 32)
      Async::Future.new { embed(documents, batch_size: batch_size).to_a }
    end

    # Generate query embeddings asynchronously
    #
    # @param queries [Array<String>, String] Query text(s) to embed
    # @param batch_size [Integer] Number of queries to process at once
    # @return [Async::Future] Future that resolves to array of SparseEmbedding objects
    def query_embed_async(queries, batch_size: 32)
      Async::Future.new { query_embed(queries, batch_size: batch_size).to_a }
    end

    # Generate passage embeddings asynchronously
    #
    # @param passages [Array<String>, String] Passage text(s) to embed
    # @param batch_size [Integer] Number of passages to process at once
    # @return [Async::Future] Future that resolves to array of SparseEmbedding objects
    def passage_embed_async(passages, batch_size: 32)
      Async::Future.new { passage_embed(passages, batch_size: batch_size).to_a }
    end

    # List all supported sparse models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_SPARSE_MODELS.values.map(&:to_h)
    end

    private

    def resolve_model_info(model_name)
      # Check built-in registry first
      info = SUPPORTED_SPARSE_MODELS[model_name]
      return info if info

      # Check custom registry
      info = CustomModelRegistry.sparse_models[model_name]
      return info if info

      raise Error, "Unknown sparse model: #{model_name}"
    end

    def create_local_model_info(model_name:, model_file:, tokenizer_file:)
      SparseModelInfo.new(
        model_name: model_name,
        description: 'Local sparse model',
        size_in_gb: 0,
        sources: {},
        model_file: model_file || 'model.onnx',
        tokenizer_file: tokenizer_file || 'tokenizer.json'
      )
    end

    def compute_sparse_embeddings(texts)
      prepared = tokenize_and_prepare(texts)
      outputs = @session.run(nil, prepared[:inputs])
      logits = extract_logits(outputs)

      # Convert to sparse embeddings
      texts.length.times.map do |i|
        create_sparse_embedding(logits[i], prepared[:attention_mask][i])
      end
    end

    def extract_logits(outputs)
      # SPLADE outputs logits for each token position
      if outputs.is_a?(Hash)
        outputs['logits'] || outputs.values.first
      else
        outputs.first
      end
    end

    # Transform token logits into sparse embedding using SPLADE algorithm
    #
    # SPLADE (Sparse Lexical AnD Expansion) uses a log-saturation function:
    # weight = log(1 + ReLU(logit))
    #
    # This ensures:
    # - ReLU removes negative activations (irrelevant terms)
    # - log1p provides saturation to prevent any term from dominating
    # - Max-pooling across positions captures the strongest signal per vocabulary term
    #
    # @param token_logits [Array<Array<Float>>] Logits for each token position
    # @param attention_mask [Array<Integer>] Mask indicating valid positions
    # @return [SparseEmbedding] Sparse vector with vocabulary indices and weights
    def create_sparse_embedding(token_logits, attention_mask)
      vocab_size = token_logits.first.length
      max_weights = Array.new(vocab_size, 0.0)

      token_logits.each_with_index do |logits, pos|
        next if attention_mask[pos].zero?

        logits.each_with_index do |logit, vocab_idx|
          # SPLADE transformation: log(1 + ReLU(x))
          activated = logit.positive? ? logit : 0.0
          weight = Math.log(1.0 + activated)
          max_weights[vocab_idx] = weight if weight > max_weights[vocab_idx]
        end
      end

      # Extract non-zero indices and values
      indices = []
      values = []

      max_weights.each_with_index do |weight, idx|
        if weight.positive?
          indices << idx
          values << weight
        end
      end

      SparseEmbedding.new(indices: indices, values: values)
    end
  end
end
