# frozen_string_literal: true

module Fastembed
  # Model information for sparse embedding models (SPLADE, etc.)
  class SparseModelInfo
    include BaseModelInfo

    def initialize(
      model_name:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'onnx/model.onnx',
      tokenizer_file: 'tokenizer.json',
      max_length: BaseModelInfo::DEFAULT_MAX_LENGTH
    )
      initialize_base(
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file,
        max_length: max_length
      )
    end

    def to_h
      {
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: tokenizer_file,
        max_length: max_length
      }
    end
  end

  # Registry of supported sparse embedding models
  SUPPORTED_SPARSE_MODELS = {
    'prithivida/Splade_PP_en_v1' => SparseModelInfo.new(
      model_name: 'prithivida/Splade_PP_en_v1',
      description: 'SPLADE++ model for sparse text retrieval',
      size_in_gb: 0.53,
      sources: { hf: 'Xenova/splade-pp-en-v1' }
    ),
    'prithvida/Splade_PP_en_v2' => SparseModelInfo.new(
      model_name: 'prithvida/Splade_PP_en_v2',
      description: 'SPLADE++ v2 with improved performance',
      size_in_gb: 0.53,
      sources: { hf: 'prithvida/Splade_PP_en_v2' }
    )
  }.freeze

  DEFAULT_SPARSE_MODEL = 'prithivida/Splade_PP_en_v1'
end
