# frozen_string_literal: true

require_relative 'fastembed/version'

module Fastembed
  class Error < StandardError; end
  class DownloadError < Error; end
end

require_relative 'fastembed/base_model_info'
require_relative 'fastembed/model_info'
require_relative 'fastembed/reranker_model_info'
require_relative 'fastembed/sparse_model_info'
require_relative 'fastembed/model_management'
require_relative 'fastembed/pooling'
require_relative 'fastembed/base_model'
require_relative 'fastembed/onnx_embedding_model'
require_relative 'fastembed/text_embedding'
require_relative 'fastembed/text_cross_encoder'
require_relative 'fastembed/sparse_embedding'
