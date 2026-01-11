# frozen_string_literal: true

module Fastembed
  # Registry for custom user-defined models
  #
  # Allows users to register arbitrary ONNX models that aren't in the built-in registry.
  # Custom models can be loaded from HuggingFace or local directories.
  #
  # @example Register a custom embedding model
  #   Fastembed.register_model(
  #     model_name: 'my-org/my-model',
  #     dim: 768,
  #     description: 'My custom model',
  #     sources: { hf: 'my-org/my-model-onnx' }
  #   )
  #   embed = Fastembed::TextEmbedding.new(model_name: 'my-org/my-model')
  #
  # @example Register a local model
  #   Fastembed.register_model(
  #     model_name: 'local-model',
  #     dim: 384,
  #     description: 'Local model',
  #     sources: {}
  #   )
  #   embed = Fastembed::TextEmbedding.new(
  #     model_name: 'local-model',
  #     local_model_dir: '/path/to/model'
  #   )
  #
  module CustomModelRegistry
    class << self
      # Custom embedding models registry
      # @return [Hash<String, ModelInfo>]
      def embedding_models
        @embedding_models ||= {}
      end

      # Custom reranker models registry
      # @return [Hash<String, RerankerModelInfo>]
      def reranker_models
        @reranker_models ||= {}
      end

      # Custom sparse models registry
      # @return [Hash<String, SparseModelInfo>]
      def sparse_models
        @sparse_models ||= {}
      end

      # Custom late interaction models registry
      # @return [Hash<String, LateInteractionModelInfo>]
      def late_interaction_models
        @late_interaction_models ||= {}
      end

      # Register a custom embedding model
      #
      # @param model_name [String] Unique model identifier
      # @param dim [Integer] Output embedding dimension
      # @param description [String] Human-readable description
      # @param sources [Hash] Source repositories (e.g., { hf: 'org/repo' })
      # @param size_in_gb [Float] Approximate model size
      # @param model_file [String] Path to ONNX file within model directory
      # @param tokenizer_file [String] Path to tokenizer.json
      # @param pooling [Symbol] Pooling strategy (:mean or :cls)
      # @param normalize [Boolean] Whether to L2 normalize outputs
      # @param max_length [Integer] Maximum sequence length
      # @return [ModelInfo] The registered model info
      def register_embedding_model(
        model_name:,
        dim:,
        description: 'Custom model',
        sources: {},
        size_in_gb: 0,
        model_file: 'model.onnx',
        tokenizer_file: 'tokenizer.json',
        pooling: :mean,
        normalize: true,
        max_length: 512
      )
        embedding_models[model_name] = ModelInfo.new(
          model_name: model_name,
          dim: dim,
          description: description,
          sources: sources,
          size_in_gb: size_in_gb,
          model_file: model_file,
          tokenizer_file: tokenizer_file,
          pooling: pooling,
          normalize: normalize,
          max_length: max_length
        )
      end

      # Register a custom reranker model
      #
      # @param model_name [String] Unique model identifier
      # @param description [String] Human-readable description
      # @param sources [Hash] Source repositories
      # @param size_in_gb [Float] Approximate model size
      # @param model_file [String] Path to ONNX file
      # @param tokenizer_file [String] Path to tokenizer.json
      # @param max_length [Integer] Maximum sequence length
      # @return [RerankerModelInfo] The registered model info
      def register_reranker_model(
        model_name:,
        description: 'Custom reranker',
        sources: {},
        size_in_gb: 0,
        model_file: 'model.onnx',
        tokenizer_file: 'tokenizer.json',
        max_length: 512
      )
        reranker_models[model_name] = RerankerModelInfo.new(
          model_name: model_name,
          description: description,
          sources: sources,
          size_in_gb: size_in_gb,
          model_file: model_file,
          tokenizer_file: tokenizer_file,
          max_length: max_length
        )
      end

      # Register a custom sparse embedding model
      #
      # @param model_name [String] Unique model identifier
      # @param description [String] Human-readable description
      # @param sources [Hash] Source repositories
      # @param size_in_gb [Float] Approximate model size
      # @param model_file [String] Path to ONNX file
      # @param tokenizer_file [String] Path to tokenizer.json
      # @param max_length [Integer] Maximum sequence length
      # @return [SparseModelInfo] The registered model info
      def register_sparse_model(
        model_name:,
        description: 'Custom sparse model',
        sources: {},
        size_in_gb: 0,
        model_file: 'model.onnx',
        tokenizer_file: 'tokenizer.json',
        max_length: 512
      )
        sparse_models[model_name] = SparseModelInfo.new(
          model_name: model_name,
          description: description,
          sources: sources,
          size_in_gb: size_in_gb,
          model_file: model_file,
          tokenizer_file: tokenizer_file,
          max_length: max_length
        )
      end

      # Register a custom late interaction model
      #
      # @param model_name [String] Unique model identifier
      # @param dim [Integer] Output embedding dimension per token
      # @param description [String] Human-readable description
      # @param sources [Hash] Source repositories
      # @param size_in_gb [Float] Approximate model size
      # @param model_file [String] Path to ONNX file
      # @param tokenizer_file [String] Path to tokenizer.json
      # @param max_length [Integer] Maximum sequence length
      # @return [LateInteractionModelInfo] The registered model info
      def register_late_interaction_model(
        model_name:,
        dim:,
        description: 'Custom late interaction model',
        sources: {},
        size_in_gb: 0,
        model_file: 'model.onnx',
        tokenizer_file: 'tokenizer.json',
        max_length: 512
      )
        late_interaction_models[model_name] = LateInteractionModelInfo.new(
          model_name: model_name,
          dim: dim,
          description: description,
          sources: sources,
          size_in_gb: size_in_gb,
          model_file: model_file,
          tokenizer_file: tokenizer_file,
          max_length: max_length
        )
      end

      # Unregister a custom model
      #
      # @param model_name [String] Model to unregister
      # @param type [Symbol] Model type (:embedding, :reranker, :sparse, :late_interaction)
      # @return [Boolean] True if model was removed
      def unregister_model(model_name, type: :embedding)
        registry = case type
                   when :embedding then embedding_models
                   when :reranker then reranker_models
                   when :sparse then sparse_models
                   when :late_interaction then late_interaction_models
                   else raise ArgumentError, "Unknown model type: #{type}"
                   end
        !registry.delete(model_name).nil?
      end

      # Clear all custom models
      # @return [void]
      def clear_all
        @embedding_models = {}
        @reranker_models = {}
        @sparse_models = {}
        @late_interaction_models = {}
      end

      # List all custom models
      # @return [Hash] All custom models by type
      def list_all
        {
          embedding: embedding_models.keys,
          reranker: reranker_models.keys,
          sparse: sparse_models.keys,
          late_interaction: late_interaction_models.keys
        }
      end
    end
  end

  # Convenience methods on the Fastembed module
  class << self
    # Register a custom embedding model
    # @see CustomModelRegistry.register_embedding_model
    def register_model(**)
      CustomModelRegistry.register_embedding_model(**)
    end

    # Register a custom reranker model
    # @see CustomModelRegistry.register_reranker_model
    def register_reranker(**)
      CustomModelRegistry.register_reranker_model(**)
    end

    # Register a custom sparse model
    # @see CustomModelRegistry.register_sparse_model
    def register_sparse_model(**)
      CustomModelRegistry.register_sparse_model(**)
    end

    # Register a custom late interaction model
    # @see CustomModelRegistry.register_late_interaction_model
    def register_late_interaction_model(**)
      CustomModelRegistry.register_late_interaction_model(**)
    end

    # List all custom registered models
    # @return [Hash] Custom models by type
    def custom_models
      CustomModelRegistry.list_all
    end
  end
end
