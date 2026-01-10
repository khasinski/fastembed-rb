# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::Quantization do
  describe '.valid?' do
    it 'returns true for valid quantization types' do
      %i[fp32 fp16 int8 uint8 q4].each do |type|
        expect(described_class.valid?(type)).to be true
      end
    end

    it 'returns false for invalid types' do
      expect(described_class.valid?(:invalid)).to be false
      expect(described_class.valid?(nil)).to be false
    end
  end

  describe '.suffix' do
    it 'returns empty string for fp32' do
      expect(described_class.suffix(:fp32)).to eq('')
    end

    it 'returns correct suffix for fp16' do
      expect(described_class.suffix(:fp16)).to eq('_fp16')
    end

    it 'returns correct suffix for int8' do
      expect(described_class.suffix(:int8)).to eq('_int8')
    end

    it 'returns correct suffix for uint8' do
      expect(described_class.suffix(:uint8)).to eq('_uint8')
    end

    it 'returns correct suffix for q4' do
      expect(described_class.suffix(:q4)).to eq('_q4')
    end
  end

  describe '.model_file' do
    it 'returns base file for fp32' do
      expect(described_class.model_file('onnx/model.onnx', :fp32)).to eq('onnx/model.onnx')
    end

    it 'returns base file for nil' do
      expect(described_class.model_file('onnx/model.onnx', nil)).to eq('onnx/model.onnx')
    end

    it 'appends suffix before extension for fp16' do
      expect(described_class.model_file('onnx/model.onnx', :fp16)).to eq('onnx/model_fp16.onnx')
    end

    it 'appends suffix before extension for int8' do
      expect(described_class.model_file('onnx/model.onnx', :int8)).to eq('onnx/model_int8.onnx')
    end

    it 'handles files without directory' do
      expect(described_class.model_file('model.onnx', :fp16)).to eq('model_fp16.onnx')
    end
  end

  describe '.list' do
    it 'returns array of quantization info' do
      list = described_class.list

      expect(list).to be_an(Array)
      expect(list.length).to eq(5)
      expect(list.first).to have_key(:type)
      expect(list.first).to have_key(:description)
      expect(list.first).to have_key(:size_multiplier)
    end
  end

  describe 'DEFAULT' do
    it 'is fp32' do
      expect(described_class::DEFAULT).to eq(:fp32)
    end
  end
end

RSpec.describe 'TextEmbedding with quantization' do
  describe '#initialize' do
    it 'accepts valid quantization type' do
      # Just test that it doesn't raise for valid type
      expect do
        Fastembed::TextEmbedding.new(quantization: :fp32)
      end.not_to raise_error
    end

    it 'raises for invalid quantization type' do
      expect do
        Fastembed::TextEmbedding.new(quantization: :invalid)
      end.to raise_error(ArgumentError, /Invalid quantization type/)
    end
  end
end
