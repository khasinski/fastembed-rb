# frozen_string_literal: true

module Fastembed
  # Model information structure
  class ModelInfo
    attr_reader :model_name, :dim, :description, :size_in_gb, :model_file,
                :tokenizer_file, :sources, :pooling, :normalize

    def initialize(
      model_name:,
      dim:,
      description:,
      size_in_gb:,
      sources:,
      model_file: "model.onnx",
      tokenizer_file: "tokenizer.json",
      pooling: :mean,
      normalize: true
    )
      @model_name = model_name
      @dim = dim
      @description = description
      @size_in_gb = size_in_gb
      @sources = sources
      @model_file = model_file
      @tokenizer_file = tokenizer_file
      @pooling = pooling
      @normalize = normalize
    end

    def hf_repo
      sources[:hf]
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
        normalize: normalize
      }
    end
  end

  # Registry of supported models
  SUPPORTED_MODELS = {
    "BAAI/bge-small-en-v1.5" => ModelInfo.new(
      model_name: "BAAI/bge-small-en-v1.5",
      dim: 384,
      description: "Fast and accurate English embedding model",
      size_in_gb: 0.067,
      sources: { hf: "Xenova/bge-small-en-v1.5" },
      model_file: "onnx/model.onnx"
    ),
    "BAAI/bge-base-en-v1.5" => ModelInfo.new(
      model_name: "BAAI/bge-base-en-v1.5",
      dim: 768,
      description: "Balanced English embedding model with higher accuracy",
      size_in_gb: 0.210,
      sources: { hf: "Xenova/bge-base-en-v1.5" },
      model_file: "onnx/model.onnx"
    ),
    "BAAI/bge-large-en-v1.5" => ModelInfo.new(
      model_name: "BAAI/bge-large-en-v1.5",
      dim: 1024,
      description: "High accuracy English embedding model",
      size_in_gb: 1.2,
      sources: { hf: "Xenova/bge-large-en-v1.5" },
      model_file: "onnx/model.onnx"
    ),
    "sentence-transformers/all-MiniLM-L6-v2" => ModelInfo.new(
      model_name: "sentence-transformers/all-MiniLM-L6-v2",
      dim: 384,
      description: "Lightweight general-purpose sentence embedding model",
      size_in_gb: 0.09,
      sources: { hf: "Xenova/all-MiniLM-L6-v2" },
      model_file: "onnx/model.onnx"
    ),
    "intfloat/multilingual-e5-small" => ModelInfo.new(
      model_name: "intfloat/multilingual-e5-small",
      dim: 384,
      description: "Multilingual embedding model supporting 100+ languages",
      size_in_gb: 0.45,
      sources: { hf: "Xenova/multilingual-e5-small" },
      model_file: "onnx/model.onnx"
    ),
    "intfloat/multilingual-e5-base" => ModelInfo.new(
      model_name: "intfloat/multilingual-e5-base",
      dim: 768,
      description: "Larger multilingual embedding model",
      size_in_gb: 1.11,
      sources: { hf: "Xenova/multilingual-e5-base" },
      model_file: "onnx/model.onnx"
    )
  }.freeze

  DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
end
