# frozen_string_literal: true

module Fastembed
  # Pooling strategies for transformer model outputs
  #
  # Transforms variable-length token embeddings into fixed-size sentence embeddings.
  # Supports mean pooling (default) and CLS token pooling.
  #
  # @example Apply mean pooling with normalization
  #   pooled = Pooling.apply(:mean, token_embeddings, attention_mask)
  #
  module Pooling
    # Valid pooling strategies
    VALID_STRATEGIES = %i[mean cls].freeze

    # Check if a pooling strategy is valid
    #
    # @param strategy [Symbol] Pooling strategy to check
    # @return [Boolean] True if valid
    def self.valid?(strategy)
      VALID_STRATEGIES.include?(strategy)
    end

    class << self
      # Mean pooling - averages all token embeddings weighted by attention mask
      #
      # @param token_embeddings [Array<Array<Array<Float>>>] Token embeddings [batch, seq, hidden]
      # @param attention_mask [Array<Array<Integer>>] Attention mask [batch, seq]
      # @return [Array<Array<Float>>] Pooled embeddings [batch, hidden]
      def mean_pooling(token_embeddings, attention_mask)
        # token_embeddings: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]

        batch_size = token_embeddings.length
        hidden_size = token_embeddings[0][0].length

        batch_size.times.map do |batch_idx|
          embeddings = token_embeddings[batch_idx]
          mask = attention_mask[batch_idx]
          seq_len = embeddings.length

          # Sum embeddings weighted by attention mask
          summed = Array.new(hidden_size, 0.0)
          mask_sum = 0.0

          seq_len.times do |seq_idx|
            weight = mask[seq_idx].to_f
            mask_sum += weight
            next if weight.zero?

            hidden_size.times do |hidden_idx|
              summed[hidden_idx] += embeddings[seq_idx][hidden_idx] * weight
            end
          end

          # Avoid division by zero
          mask_sum = 1.0 if mask_sum.zero?

          # Divide by sum of mask to get mean
          summed.map { |v| v / mask_sum }
        end
      end

      # CLS pooling - uses the [CLS] token embedding (first token)
      #
      # @param token_embeddings [Array<Array<Array<Float>>>] Token embeddings [batch, seq, hidden]
      # @param _attention_mask [Array<Array<Integer>>] Attention mask (unused)
      # @return [Array<Array<Float>>] Pooled embeddings [batch, hidden]
      def cls_pooling(token_embeddings, _attention_mask)
        token_embeddings.map { |batch| batch[0] }
      end

      # L2 normalize vectors to unit length
      #
      # @param vectors [Array<Array<Float>>] Vectors to normalize
      # @return [Array<Array<Float>>] Normalized vectors
      def normalize(vectors)
        vectors.map do |vector|
          norm = Math.sqrt(vector.sum { |v| v * v })
          norm = 1.0 if norm.zero?
          vector.map { |v| v / norm }
        end
      end

      # Apply pooling strategy to token embeddings
      #
      # @param strategy [Symbol] Pooling strategy (:mean or :cls)
      # @param token_embeddings [Array] Token embeddings from model
      # @param attention_mask [Array] Attention mask
      # @param should_normalize [Boolean] Whether to L2 normalize output
      # @return [Array<Array<Float>>] Pooled embeddings
      # @raise [ArgumentError] If unknown pooling strategy
      def apply(strategy, token_embeddings, attention_mask, should_normalize: true)
        pooled = case strategy
                 when :mean
                   mean_pooling(token_embeddings, attention_mask)
                 when :cls
                   cls_pooling(token_embeddings, attention_mask)
                 else
                   raise ArgumentError, "Unknown pooling strategy: #{strategy}"
                 end

        should_normalize ? normalize(pooled) : pooled
      end
    end
  end
end
