# frozen_string_literal: true

module Fastembed
  # Progress information for batch operations
  # Passed to progress callbacks during embedding
  class Progress
    attr_reader :current, :total, :batch_size

    # @param current [Integer] Current batch number (1-indexed)
    # @param total [Integer] Total number of batches
    # @param batch_size [Integer] Size of each batch
    def initialize(current:, total:, batch_size:)
      @current = current
      @total = total
      @batch_size = batch_size
    end

    # Percentage complete (0.0 to 1.0)
    # @return [Float]
    def percentage
      return 1.0 if total.zero?

      current.to_f / total
    end

    # Percentage as integer (0 to 100)
    # @return [Integer]
    def percent
      (percentage * 100).round
    end

    # Number of documents processed so far
    # @return [Integer]
    def documents_processed
      current * batch_size
    end

    # Check if processing is complete
    # @return [Boolean]
    def complete?
      current >= total
    end

    def to_s
      "Progress(#{current}/#{total}, #{percent}%)"
    end

    def inspect
      to_s
    end
  end
end
