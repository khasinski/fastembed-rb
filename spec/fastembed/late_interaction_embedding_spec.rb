# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::LateInteractionEmbedding do
  describe '#initialize' do
    it 'stores token embeddings' do
      embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
      emb = described_class.new(embeddings)

      expect(emb.embeddings).to eq(embeddings)
      expect(emb.token_count).to eq(3)
    end
  end

  describe '#dim' do
    it 'returns embedding dimension' do
      emb = described_class.new([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

      expect(emb.dim).to eq(3)
    end

    it 'returns 0 for empty embeddings' do
      emb = described_class.new([])

      expect(emb.dim).to eq(0)
    end
  end

  describe '#max_sim' do
    it 'computes MaxSim score between query and document' do
      # Simple case: 2 query tokens, 2 doc tokens
      query = described_class.new([[1.0, 0.0], [0.0, 1.0]])
      doc = described_class.new([[0.8, 0.6], [0.6, 0.8]])

      # MaxSim: for each query token, find max dot product with doc tokens
      # Query[0] = [1,0]: max(0.8, 0.6) = 0.8
      # Query[1] = [0,1]: max(0.6, 0.8) = 0.8
      # Total = 1.6
      score = query.max_sim(doc)

      expect(score).to be_within(0.001).of(1.6)
    end

    it 'returns 0 for empty embeddings' do
      empty = described_class.new([])
      non_empty = described_class.new([[1.0, 0.0]])

      expect(empty.max_sim(non_empty)).to eq(0.0)
      expect(non_empty.max_sim(empty)).to eq(0.0)
    end
  end

  describe '#to_a' do
    it 'returns embeddings as array' do
      embeddings = [[0.1, 0.2], [0.3, 0.4]]
      emb = described_class.new(embeddings)

      expect(emb.to_a).to eq(embeddings)
    end
  end

  describe '#to_s' do
    it 'returns readable string representation' do
      emb = described_class.new([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

      expect(emb.to_s).to eq('LateInteractionEmbedding(tokens=2, dim=3)')
    end
  end
end

RSpec.describe Fastembed::LateInteractionModelInfo do
  describe 'SUPPORTED_LATE_INTERACTION_MODELS' do
    it 'contains the default model' do
      expect(Fastembed::SUPPORTED_LATE_INTERACTION_MODELS).to have_key(Fastembed::DEFAULT_LATE_INTERACTION_MODEL)
    end

    it 'has HuggingFace source for all models' do
      Fastembed::SUPPORTED_LATE_INTERACTION_MODELS.each do |name, info|
        expect(info.hf_repo).not_to be_nil, "Model #{name} missing HF repo"
      end
    end

    it 'has required attributes for all models' do
      Fastembed::SUPPORTED_LATE_INTERACTION_MODELS.each_value do |info|
        expect(info.model_name).to be_a(String)
        expect(info.dim).to be_a(Integer)
        expect(info.description).to be_a(String)
        expect(info.size_in_gb).to be_a(Numeric)
      end
    end
  end

  describe '#to_h' do
    it 'converts model info to hash' do
      model = Fastembed::SUPPORTED_LATE_INTERACTION_MODELS['colbert-ir/colbertv2.0']
      hash = model.to_h

      expect(hash[:model_name]).to eq('colbert-ir/colbertv2.0')
      expect(hash[:dim]).to eq(128)
      expect(hash).to have_key(:description)
    end
  end
end

RSpec.describe Fastembed::LateInteractionTextEmbedding do
  describe '.list_supported_models' do
    it 'returns array of model hashes' do
      models = described_class.list_supported_models

      expect(models).to be_an(Array)
      expect(models).not_to be_empty
      expect(models.first).to have_key(:model_name)
      expect(models.first).to have_key(:dim)
    end
  end
end
