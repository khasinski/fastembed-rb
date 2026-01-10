# frozen_string_literal: true

module Fastembed
  # Shared functionality for model information classes
  module BaseModelInfo
    attr_reader :model_name, :description, :size_in_gb, :model_file,
                :tokenizer_file, :sources

    # Returns the HuggingFace repository ID
    # @return [String] The HF repo ID
    def hf_repo
      sources[:hf]
    end

    private

    def initialize_base(model_name:, description:, size_in_gb:, sources:, model_file:, tokenizer_file:)
      @model_name = model_name
      @description = description
      @size_in_gb = size_in_gb
      @sources = sources
      @model_file = model_file
      @tokenizer_file = tokenizer_file
    end
  end
end
