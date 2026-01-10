# frozen_string_literal: true

module Fastembed
  # Input validation helpers for embedding models
  #
  # Provides consistent validation and normalization of documents, queries,
  # and other inputs across all model types.
  #
  # @api private
  #
  module Validators
    class << self
      # Validate and normalize document input
      #
      # Ensures documents are not nil, converts single strings to arrays,
      # and validates that no individual document is nil.
      #
      # @param documents [Array<String>, String, nil] Documents to validate
      # @return [Array<String>] Normalized array of documents
      # @raise [ArgumentError] If documents is nil or contains nil values
      def validate_documents!(documents)
        raise ArgumentError, 'documents cannot be nil' if documents.nil?

        documents = [documents] if documents.is_a?(String)

        documents.each_with_index do |doc, i|
          raise ArgumentError, "document at index #{i} cannot be nil" if doc.nil?
        end

        documents
      end

      # Validate query and documents for reranking
      #
      # @param query [String, nil] Query to validate
      # @param documents [Array<String>, nil] Documents to validate
      # @return [Array<String>] Validated documents array
      # @raise [ArgumentError] If query or documents is nil, or documents contains nil
      def validate_rerank_input!(query:, documents:)
        raise ArgumentError, 'query cannot be nil' if query.nil?
        raise ArgumentError, 'documents cannot be nil' if documents.nil?

        documents.each_with_index do |doc, i|
          raise ArgumentError, "document at index #{i} cannot be nil" if doc.nil?
        end

        documents
      end

      # Check if documents array is empty
      #
      # @param documents [Array<String>] Documents to check
      # @return [Boolean] True if empty
      def empty?(documents)
        documents.empty?
      end
    end
  end
end
