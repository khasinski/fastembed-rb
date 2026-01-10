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
    def initialize(
      model_name: DEFAULT_SPARSE_MODEL,
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

      load_model
      load_tokenizer
    end

    # Generate sparse embeddings for documents
    #
    # @param documents [Array<String>, String] Text document(s) to embed
    # @param batch_size [Integer] Number of documents to process at once
    # @return [Enumerator] Lazy enumerator yielding SparseEmbedding objects
    def embed(documents, batch_size: 32)
      raise ArgumentError, 'documents cannot be nil' if documents.nil?

      documents = [documents] if documents.is_a?(String)
      return Enumerator.new { |_| } if documents.empty?

      documents.each_with_index do |doc, i|
        raise ArgumentError, "document at index #{i} cannot be nil" if doc.nil?
      end

      Enumerator.new do |yielder|
        documents.each_slice(batch_size) do |batch|
          sparse_embeddings = compute_sparse_embeddings(batch)
          sparse_embeddings.each { |emb| yielder << emb }
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

    # List all supported sparse models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_SPARSE_MODELS.values.map(&:to_h)
    end

    private

    def resolve_model_info(model_name)
      info = SUPPORTED_SPARSE_MODELS[model_name]
      raise Error, "Unknown sparse model: #{model_name}" unless info

      info
    end

    def load_model
      model_path = File.join(@model_dir, @model_info.model_file)
      @session = load_onnx_session(model_path, providers: @providers)
    end

    def load_tokenizer
      tokenizer_path = File.join(@model_dir, @model_info.tokenizer_file)
      @tokenizer = load_tokenizer_from_file(tokenizer_path, max_length: @model_info.max_length)
    end

    def compute_sparse_embeddings(texts)
      # Tokenize
      encodings = @tokenizer.encode_batch(texts)

      # Prepare inputs
      input_ids = encodings.map(&:ids)
      attention_mask = encodings.map(&:attention_mask)
      token_type_ids = encodings.map { |e| e.type_ids || Array.new(e.ids.length, 0) }

      inputs = {
        'input_ids' => input_ids,
        'attention_mask' => attention_mask
      }
      inputs['token_type_ids'] = token_type_ids if input_names.include?('token_type_ids')

      # Run inference
      outputs = @session.run(nil, inputs)
      logits = extract_logits(outputs)

      # Convert to sparse embeddings
      texts.length.times.map do |i|
        create_sparse_embedding(logits[i], attention_mask[i])
      end
    end

    def input_names
      @input_names ||= @session.inputs.map { |i| i[:name] }
    end

    def extract_logits(outputs)
      # SPLADE outputs logits for each token position
      if outputs.is_a?(Hash)
        outputs['logits'] || outputs.values.first
      else
        outputs.first
      end
    end

    def create_sparse_embedding(token_logits, attention_mask)
      # Apply SPLADE transformation: log(1 + ReLU(x))
      # and max-pool across sequence positions
      vocab_size = token_logits.first.length
      max_weights = Array.new(vocab_size, 0.0)

      token_logits.each_with_index do |logits, pos|
        next if attention_mask[pos].zero?

        logits.each_with_index do |logit, vocab_idx|
          # ReLU
          activated = logit > 0 ? logit : 0.0
          # log(1 + x)
          weight = Math.log(1.0 + activated)
          max_weights[vocab_idx] = weight if weight > max_weights[vocab_idx]
        end
      end

      # Extract non-zero indices and values
      indices = []
      values = []

      max_weights.each_with_index do |weight, idx|
        if weight > 0
          indices << idx
          values << weight
        end
      end

      SparseEmbedding.new(indices: indices, values: values)
    end
  end
end
