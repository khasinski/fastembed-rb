# frozen_string_literal: true

module Fastembed
  # Model information for dense embedding models
  #
  # Stores metadata and configuration for ONNX embedding models including
  # output dimensions, pooling strategy, and normalization settings.
  #
  # @example Access model info
  #   info = Fastembed::SUPPORTED_MODELS['BAAI/bge-small-en-v1.5']
  #   info.dim        # => 384
  #   info.pooling    # => :mean
  #   info.normalize  # => true
  #
  class ModelInfo
    include BaseModelInfo

    # @!attribute [r] dim
    #   @return [Integer] Output embedding dimension
    # @!attribute [r] pooling
    #   @return [Symbol] Pooling strategy (:mean or :cls)
    # @!attribute [r] normalize
    #   @return [Boolean] Whether to L2 normalize output embeddings
    attr_reader :dim, :pooling, :normalize

    # Create a new ModelInfo instance
    #
    # @param model_name [String] Full model identifier
    # @param dim [Integer] Output embedding dimension
    # @param description [String] Human-readable description
    # @param size_in_gb [Float] Model size in GB
    # @param sources [Hash] Source repositories
    # @param model_file [String] Path to ONNX model file
    # @param tokenizer_file [String] Path to tokenizer file
    # @param pooling [Symbol] Pooling strategy (:mean or :cls)
    # @param normalize [Boolean] Whether to L2 normalize outputs
    # @param max_length [Integer] Maximum sequence length
    # @raise [ArgumentError] If pooling strategy is invalid
    def initialize(
      model_name:,
      dim:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'model.onnx',
      tokenizer_file: 'tokenizer.json',
      pooling: :mean,
      normalize: true,
      max_length: BaseModelInfo::DEFAULT_MAX_LENGTH
    )
      unless Pooling.valid?(pooling)
        valid = Pooling::VALID_STRATEGIES.join(', ')
        raise ArgumentError, "Invalid pooling strategy: #{pooling}. Valid strategies: #{valid}"
      end

      initialize_base(
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file,
        max_length: max_length
      )
      @dim = dim
      @pooling = pooling
      @normalize = normalize
    end

    def to_h
      {
        model_name: model_name,
        dim: dim,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file,
        pooling: pooling,
        normalize: normalize,
        max_length: max_length
      }
    end
  end

  # Registry of supported models
  SUPPORTED_MODELS = {
    'BAAI/bge-small-en-v1.5' => ModelInfo.new(
      model_name: 'BAAI/bge-small-en-v1.5',
      dim: 384,
      description: 'Fast and accurate English embedding model',
      size_in_gb: 0.067,
      sources: { hf: 'Xenova/bge-small-en-v1.5' },
      model_file: 'onnx/model.onnx'
    ),
    'BAAI/bge-base-en-v1.5' => ModelInfo.new(
      model_name: 'BAAI/bge-base-en-v1.5',
      dim: 768,
      description: 'Balanced English embedding model with higher accuracy',
      size_in_gb: 0.210,
      sources: { hf: 'Xenova/bge-base-en-v1.5' },
      model_file: 'onnx/model.onnx'
    ),
    'BAAI/bge-large-en-v1.5' => ModelInfo.new(
      model_name: 'BAAI/bge-large-en-v1.5',
      dim: 1024,
      description: 'High accuracy English embedding model',
      size_in_gb: 1.2,
      sources: { hf: 'Xenova/bge-large-en-v1.5' },
      model_file: 'onnx/model.onnx'
    ),
    'sentence-transformers/all-MiniLM-L6-v2' => ModelInfo.new(
      model_name: 'sentence-transformers/all-MiniLM-L6-v2',
      dim: 384,
      description: 'Lightweight general-purpose sentence embedding model',
      size_in_gb: 0.09,
      sources: { hf: 'Xenova/all-MiniLM-L6-v2' },
      model_file: 'onnx/model.onnx'
    ),
    'intfloat/multilingual-e5-small' => ModelInfo.new(
      model_name: 'intfloat/multilingual-e5-small',
      dim: 384,
      description: 'Multilingual embedding model supporting 100+ languages',
      size_in_gb: 0.45,
      sources: { hf: 'Xenova/multilingual-e5-small' },
      model_file: 'onnx/model.onnx'
    ),
    'intfloat/multilingual-e5-base' => ModelInfo.new(
      model_name: 'intfloat/multilingual-e5-base',
      dim: 768,
      description: 'Larger multilingual embedding model',
      size_in_gb: 1.11,
      sources: { hf: 'Xenova/multilingual-e5-base' },
      model_file: 'onnx/model.onnx'
    ),
    'nomic-ai/nomic-embed-text-v1' => ModelInfo.new(
      model_name: 'nomic-ai/nomic-embed-text-v1',
      dim: 768,
      description: 'Long context (8192 tokens) English embedding model',
      size_in_gb: 0.52,
      sources: { hf: 'nomic-ai/nomic-embed-text-v1' },
      model_file: 'onnx/model.onnx',
      max_length: 8192
    ),
    'nomic-ai/nomic-embed-text-v1.5' => ModelInfo.new(
      model_name: 'nomic-ai/nomic-embed-text-v1.5',
      dim: 768,
      description: 'Improved long context embedding with Matryoshka support',
      size_in_gb: 0.52,
      sources: { hf: 'nomic-ai/nomic-embed-text-v1.5' },
      model_file: 'onnx/model.onnx',
      max_length: 8192
    ),
    'jinaai/jina-embeddings-v2-small-en' => ModelInfo.new(
      model_name: 'jinaai/jina-embeddings-v2-small-en',
      dim: 512,
      description: 'Small English embedding with 8192 token context',
      size_in_gb: 0.06,
      sources: { hf: 'Xenova/jina-embeddings-v2-small-en' },
      model_file: 'onnx/model.onnx',
      max_length: 8192
    ),
    'jinaai/jina-embeddings-v2-base-en' => ModelInfo.new(
      model_name: 'jinaai/jina-embeddings-v2-base-en',
      dim: 768,
      description: 'Base English embedding with 8192 token context',
      size_in_gb: 0.52,
      sources: { hf: 'Xenova/jina-embeddings-v2-base-en' },
      model_file: 'onnx/model.onnx',
      max_length: 8192
    ),
    'sentence-transformers/paraphrase-MiniLM-L6-v2' => ModelInfo.new(
      model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2',
      dim: 384,
      description: 'Optimized for paraphrase detection and semantic similarity',
      size_in_gb: 0.09,
      sources: { hf: 'Xenova/paraphrase-MiniLM-L6-v2' },
      model_file: 'onnx/model.onnx'
    ),
    'sentence-transformers/all-mpnet-base-v2' => ModelInfo.new(
      model_name: 'sentence-transformers/all-mpnet-base-v2',
      dim: 768,
      description: 'High quality general-purpose sentence embeddings',
      size_in_gb: 0.44,
      sources: { hf: 'Xenova/all-mpnet-base-v2' },
      model_file: 'onnx/model.onnx'
    )
  }.freeze

  DEFAULT_MODEL = 'BAAI/bge-small-en-v1.5'
end
