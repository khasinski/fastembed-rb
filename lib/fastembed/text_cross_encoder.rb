# frozen_string_literal: true

module Fastembed
  # Cross-encoder model for reranking query-document pairs
  #
  # Unlike embedding models that encode texts independently, cross-encoders
  # process query-document pairs together to produce relevance scores.
  # This is more accurate but slower (O(n) comparisons vs O(1) with embeddings).
  #
  # @example Basic reranking
  #   reranker = Fastembed::TextCrossEncoder.new
  #   scores = reranker.rerank(
  #     query: "What is machine learning?",
  #     documents: ["ML is a subset of AI...", "The weather is nice today"]
  #   )
  #   # => [0.95, 0.02]
  #
  # @example Get ranked results
  #   results = reranker.rerank_with_scores(
  #     query: "What is Ruby?",
  #     documents: documents
  #   )
  #   # => [{document: "Ruby is a programming...", score: 0.89, index: 2}, ...]
  #
  class TextCrossEncoder
    include BaseModel

    # Initialize a cross-encoder model for reranking
    #
    # @param model_name [String] Name of the model to use
    # @param cache_dir [String, nil] Custom cache directory for models
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers
    def initialize(
      model_name: DEFAULT_RERANKER_MODEL,
      cache_dir: nil,
      threads: nil,
      providers: nil
    )
      initialize_model(
        model_name: model_name,
        cache_dir: cache_dir,
        threads: threads,
        providers: providers,
        show_progress: true
      )

      load_model
      load_tokenizer
    end

    # Score query-document pairs and return relevance scores
    #
    # @param query [String] The query text
    # @param documents [Array<String>] Documents to score against the query
    # @param batch_size [Integer] Number of pairs to process at once
    # @return [Array<Float>] Relevance scores for each document (higher = more relevant)
    # @raise [ArgumentError] If query or documents is nil, or documents contains nil
    def rerank(query:, documents:, batch_size: 64)
      raise ArgumentError, 'query cannot be nil' if query.nil?
      raise ArgumentError, 'documents cannot be nil' if documents.nil?

      return [] if documents.empty?

      documents.each_with_index do |doc, i|
        raise ArgumentError, "document at index #{i} cannot be nil" if doc.nil?
      end

      scores = []
      documents.each_slice(batch_size) do |batch|
        batch_scores = score_pairs(query, batch)
        scores.concat(batch_scores)
      end
      scores
    end

    # Rerank documents and return sorted results with scores
    #
    # @param query [String] The query text
    # @param documents [Array<String>] Documents to rerank
    # @param top_k [Integer, nil] Return only top K results (nil = all)
    # @param batch_size [Integer] Number of pairs to process at once
    # @return [Array<Hash>] Sorted results with :document, :score, :index keys
    def rerank_with_scores(query:, documents:, top_k: nil, batch_size: 64)
      scores = rerank(query: query, documents: documents, batch_size: batch_size)

      results = documents.zip(scores).each_with_index.map do |(doc, score), idx|
        { document: doc, score: score, index: idx }
      end

      results.sort_by! { |r| -r[:score] }
      top_k ? results.first(top_k) : results
    end

    # List all supported reranker models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_RERANKER_MODELS.values.map(&:to_h)
    end

    private

    def resolve_model_info(model_name)
      info = SUPPORTED_RERANKER_MODELS[model_name]
      raise Error, "Unknown reranker model: #{model_name}" unless info

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

    def score_pairs(query, documents)
      encodings = tokenize_pairs(query, documents)
      inputs = prepare_model_inputs(encodings)
      extract_scores(@session.run(nil, inputs))
    end

    def tokenize_pairs(query, documents)
      documents.map { |doc| @tokenizer.encode(query, doc) }
    end

    def prepare_model_inputs(encodings)
      max_len = encodings.map { |e| e.ids.length }.max

      input_ids = []
      attention_mask = []
      token_type_ids = []

      encodings.each do |encoding|
        pad_len = max_len - encoding.ids.length
        input_ids << pad_sequence(encoding.ids, pad_len)
        attention_mask << pad_sequence(encoding.attention_mask, pad_len)
        token_type_ids << pad_sequence(encoding.type_ids, pad_len)
      end

      { 'input_ids' => input_ids, 'attention_mask' => attention_mask, 'token_type_ids' => token_type_ids }
    end

    def pad_sequence(sequence, pad_len)
      sequence + ([0] * pad_len)
    end

    def extract_scores(outputs)
      outputs.first.map { |logit| logit.is_a?(Array) ? logit.first : logit }
    end
  end
end
