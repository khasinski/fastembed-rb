# frozen_string_literal: true

module Fastembed
  # Quantization options for ONNX models
  # Different quantization levels trade off model size/speed vs accuracy
  module Quantization
    # Available quantization types
    TYPES = {
      fp32: {
        suffix: '',
        description: 'Full precision (32-bit float)',
        size_multiplier: 1.0
      },
      fp16: {
        suffix: '_fp16',
        description: 'Half precision (16-bit float)',
        size_multiplier: 0.5
      },
      int8: {
        suffix: '_int8',
        description: 'Dynamic int8 quantization',
        size_multiplier: 0.25
      },
      uint8: {
        suffix: '_uint8',
        description: 'Dynamic uint8 quantization',
        size_multiplier: 0.25
      },
      q4: {
        suffix: '_q4',
        description: '4-bit quantization',
        size_multiplier: 0.125
      }
    }.freeze

    # Default quantization type
    DEFAULT = :fp32

    class << self
      # Check if a quantization type is valid
      # @param type [Symbol] Quantization type
      # @return [Boolean]
      def valid?(type)
        TYPES.key?(type)
      end

      # Get the model file suffix for a quantization type
      # @param type [Symbol] Quantization type
      # @return [String] Suffix to append to model filename
      def suffix(type)
        TYPES.dig(type, :suffix) || ''
      end

      # Get quantized model filename
      # @param base_file [String] Base model file (e.g., "onnx/model.onnx")
      # @param type [Symbol] Quantization type
      # @return [String] Quantized model filename
      def model_file(base_file, type)
        return base_file if type == :fp32 || type.nil?

        ext = File.extname(base_file)
        base = base_file.chomp(ext)
        "#{base}#{suffix(type)}#{ext}"
      end

      # List available quantization types
      # @return [Array<Hash>] Array of quantization info
      def list
        TYPES.map do |type, info|
          { type: type, description: info[:description], size_multiplier: info[:size_multiplier] }
        end
      end
    end
  end
end
