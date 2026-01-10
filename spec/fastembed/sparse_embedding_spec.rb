# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::SparseEmbedding do
  describe '#initialize' do
    it 'stores indices and values' do
      emb = described_class.new(indices: [1, 5, 10], values: [0.5, 1.2, 0.8])

      expect(emb.indices).to eq([1, 5, 10])
      expect(emb.values).to eq([0.5, 1.2, 0.8])
    end
  end

  describe '#to_h' do
    it 'converts to hash format' do
      emb = described_class.new(indices: [1, 5, 10], values: [0.5, 1.2, 0.8])

      expect(emb.to_h).to eq({ 1 => 0.5, 5 => 1.2, 10 => 0.8 })
    end
  end

  describe '#nnz' do
    it 'returns number of non-zero elements' do
      emb = described_class.new(indices: [1, 5, 10], values: [0.5, 1.2, 0.8])

      expect(emb.nnz).to eq(3)
    end
  end

  describe '#to_s' do
    it 'returns readable string representation' do
      emb = described_class.new(indices: [1, 5, 10], values: [0.5, 1.2, 0.8])

      expect(emb.to_s).to eq('SparseEmbedding(nnz=3)')
    end
  end
end

RSpec.describe Fastembed::SparseModelInfo do
  describe 'SUPPORTED_SPARSE_MODELS' do
    it 'contains the default sparse model' do
      expect(Fastembed::SUPPORTED_SPARSE_MODELS).to have_key(Fastembed::DEFAULT_SPARSE_MODEL)
    end

    it 'has HuggingFace source for all models' do
      Fastembed::SUPPORTED_SPARSE_MODELS.each do |name, info|
        expect(info.hf_repo).not_to be_nil, "Model #{name} missing HF repo"
      end
    end

    it 'has required attributes for all models' do
      Fastembed::SUPPORTED_SPARSE_MODELS.each_value do |info|
        expect(info.model_name).to be_a(String)
        expect(info.description).to be_a(String)
        expect(info.size_in_gb).to be_a(Numeric)
        expect(info.model_file).to be_a(String)
        expect(info.tokenizer_file).to be_a(String)
      end
    end
  end

  describe '#to_h' do
    it 'converts model info to hash' do
      model = Fastembed::SUPPORTED_SPARSE_MODELS['prithivida/Splade_PP_en_v1']
      hash = model.to_h

      expect(hash[:model_name]).to eq('prithivida/Splade_PP_en_v1')
      expect(hash).to have_key(:description)
      expect(hash).to have_key(:size_in_gb)
    end
  end
end

RSpec.describe Fastembed::TextSparseEmbedding do
  describe '.list_supported_models' do
    it 'returns array of model hashes' do
      models = described_class.list_supported_models

      expect(models).to be_an(Array)
      expect(models).not_to be_empty
      expect(models.first).to have_key(:model_name)
    end
  end
end
