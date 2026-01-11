# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::CustomModelRegistry do
  # Clear custom registries after each test
  after do
    described_class.clear_all
  end

  describe '.embedding_models' do
    it 'returns empty hash initially' do
      expect(described_class.embedding_models).to eq({})
    end

    it 'returns registered models' do
      described_class.register_embedding_model(
        model_name: 'test/model',
        dim: 384
      )
      expect(described_class.embedding_models).to have_key('test/model')
    end
  end

  describe '.register_embedding_model' do
    it 'registers a custom embedding model' do
      info = described_class.register_embedding_model(
        model_name: 'my-org/my-model',
        dim: 768,
        description: 'My custom model',
        sources: { hf: 'my-org/my-model-onnx' }
      )

      expect(info).to be_a(Fastembed::ModelInfo)
      expect(info.model_name).to eq('my-org/my-model')
      expect(info.dim).to eq(768)
      expect(info.description).to eq('My custom model')
      expect(info.sources[:hf]).to eq('my-org/my-model-onnx')
    end

    it 'uses default values for optional parameters' do
      info = described_class.register_embedding_model(
        model_name: 'test/model',
        dim: 384
      )

      expect(info.description).to eq('Custom model')
      expect(info.size_in_gb).to eq(0)
      # Default model_file depends on model type
      expect(info.tokenizer_file).to eq('tokenizer.json')
      expect(info.pooling).to eq(:mean)
      expect(info.normalize).to be true
      expect(info.max_length).to eq(512)
    end

    it 'allows custom pooling and normalization settings' do
      info = described_class.register_embedding_model(
        model_name: 'test/cls-model',
        dim: 768,
        pooling: :cls,
        normalize: false
      )

      expect(info.pooling).to eq(:cls)
      expect(info.normalize).to be false
    end

    it 'allows custom max_length' do
      info = described_class.register_embedding_model(
        model_name: 'test/long-model',
        dim: 768,
        max_length: 8192
      )

      expect(info.max_length).to eq(8192)
    end
  end

  describe '.register_reranker_model' do
    it 'registers a custom reranker model' do
      info = described_class.register_reranker_model(
        model_name: 'my-org/my-reranker',
        description: 'My reranker',
        sources: { hf: 'my-org/my-reranker-onnx' }
      )

      expect(info).to be_a(Fastembed::RerankerModelInfo)
      expect(info.model_name).to eq('my-org/my-reranker')
      expect(described_class.reranker_models).to have_key('my-org/my-reranker')
    end

    it 'uses default values for optional parameters' do
      info = described_class.register_reranker_model(
        model_name: 'test/reranker'
      )

      expect(info.description).to eq('Custom reranker')
      expect(info.model_file).to eq('onnx/model.onnx')
      expect(info.tokenizer_file).to eq('tokenizer.json')
    end
  end

  describe '.register_sparse_model' do
    it 'registers a custom sparse model' do
      info = described_class.register_sparse_model(
        model_name: 'my-org/my-sparse',
        description: 'My sparse model',
        sources: { hf: 'my-org/my-sparse-onnx' }
      )

      expect(info).to be_a(Fastembed::SparseModelInfo)
      expect(info.model_name).to eq('my-org/my-sparse')
      expect(described_class.sparse_models).to have_key('my-org/my-sparse')
    end
  end

  describe '.register_late_interaction_model' do
    it 'registers a custom late interaction model' do
      info = described_class.register_late_interaction_model(
        model_name: 'my-org/my-colbert',
        dim: 128,
        description: 'My ColBERT model',
        sources: { hf: 'my-org/my-colbert-onnx' }
      )

      expect(info).to be_a(Fastembed::LateInteractionModelInfo)
      expect(info.model_name).to eq('my-org/my-colbert')
      expect(info.dim).to eq(128)
      expect(described_class.late_interaction_models).to have_key('my-org/my-colbert')
    end
  end

  describe '.unregister_model' do
    before do
      described_class.register_embedding_model(model_name: 'test/embed', dim: 384)
      described_class.register_reranker_model(model_name: 'test/rerank')
      described_class.register_sparse_model(model_name: 'test/sparse')
      described_class.register_late_interaction_model(model_name: 'test/colbert', dim: 128)
    end

    it 'removes embedding model' do
      result = described_class.unregister_model('test/embed', type: :embedding)

      expect(result).to be true
      expect(described_class.embedding_models).not_to have_key('test/embed')
    end

    it 'removes reranker model' do
      result = described_class.unregister_model('test/rerank', type: :reranker)

      expect(result).to be true
      expect(described_class.reranker_models).not_to have_key('test/rerank')
    end

    it 'removes sparse model' do
      result = described_class.unregister_model('test/sparse', type: :sparse)

      expect(result).to be true
      expect(described_class.sparse_models).not_to have_key('test/sparse')
    end

    it 'removes late interaction model' do
      result = described_class.unregister_model('test/colbert', type: :late_interaction)

      expect(result).to be true
      expect(described_class.late_interaction_models).not_to have_key('test/colbert')
    end

    it 'returns false for non-existent model' do
      result = described_class.unregister_model('nonexistent', type: :embedding)

      expect(result).to be false
    end

    it 'raises error for unknown type' do
      expect do
        described_class.unregister_model('test', type: :unknown)
      end.to raise_error(ArgumentError, /Unknown model type/)
    end
  end

  describe '.clear_all' do
    it 'clears all custom registries' do
      described_class.register_embedding_model(model_name: 'test/embed', dim: 384)
      described_class.register_reranker_model(model_name: 'test/rerank')
      described_class.register_sparse_model(model_name: 'test/sparse')
      described_class.register_late_interaction_model(model_name: 'test/colbert', dim: 128)

      described_class.clear_all

      expect(described_class.embedding_models).to be_empty
      expect(described_class.reranker_models).to be_empty
      expect(described_class.sparse_models).to be_empty
      expect(described_class.late_interaction_models).to be_empty
    end
  end

  describe '.list_all' do
    it 'lists all custom models by type' do
      described_class.register_embedding_model(model_name: 'test/embed1', dim: 384)
      described_class.register_embedding_model(model_name: 'test/embed2', dim: 768)
      described_class.register_reranker_model(model_name: 'test/rerank')

      result = described_class.list_all

      expect(result[:embedding]).to contain_exactly('test/embed1', 'test/embed2')
      expect(result[:reranker]).to contain_exactly('test/rerank')
      expect(result[:sparse]).to be_empty
      expect(result[:late_interaction]).to be_empty
    end
  end
end

RSpec.describe 'Fastembed module convenience methods' do
  after do
    Fastembed::CustomModelRegistry.clear_all
  end

  describe '.register_model' do
    it 'registers embedding model through Fastembed module' do
      Fastembed.register_model(
        model_name: 'test/via-module',
        dim: 512
      )

      expect(Fastembed::CustomModelRegistry.embedding_models).to have_key('test/via-module')
    end
  end

  describe '.register_reranker' do
    it 'registers reranker through Fastembed module' do
      Fastembed.register_reranker(model_name: 'test/reranker')

      expect(Fastembed::CustomModelRegistry.reranker_models).to have_key('test/reranker')
    end
  end

  describe '.register_sparse_model' do
    it 'registers sparse model through Fastembed module' do
      Fastembed.register_sparse_model(model_name: 'test/sparse')

      expect(Fastembed::CustomModelRegistry.sparse_models).to have_key('test/sparse')
    end
  end

  describe '.register_late_interaction_model' do
    it 'registers late interaction model through Fastembed module' do
      Fastembed.register_late_interaction_model(model_name: 'test/colbert', dim: 128)

      expect(Fastembed::CustomModelRegistry.late_interaction_models).to have_key('test/colbert')
    end
  end

  describe '.custom_models' do
    it 'returns all custom models' do
      Fastembed.register_model(model_name: 'test/embed', dim: 384)
      Fastembed.register_reranker(model_name: 'test/rerank')

      result = Fastembed.custom_models

      expect(result[:embedding]).to include('test/embed')
      expect(result[:reranker]).to include('test/rerank')
    end
  end
end

RSpec.describe 'Custom model integration with TextEmbedding' do
  after do
    Fastembed::CustomModelRegistry.clear_all
  end

  describe 'TextEmbedding.list_supported_models' do
    it 'includes custom models in the list' do
      Fastembed.register_model(
        model_name: 'custom/test-model',
        dim: 512,
        description: 'A custom test model'
      )

      models = Fastembed::TextEmbedding.list_supported_models

      custom = models.find { |m| m[:model_name] == 'custom/test-model' }
      expect(custom).not_to be_nil
      expect(custom[:dim]).to eq(512)
    end
  end

  describe 'TextEmbedding.get_model_info' do
    it 'returns info for custom model' do
      Fastembed.register_model(
        model_name: 'custom/test-model',
        dim: 256
      )

      info = Fastembed::TextEmbedding.get_model_info('custom/test-model')

      expect(info).not_to be_nil
      expect(info[:dim]).to eq(256)
    end
  end

  describe 'ModelManagement.resolve_model_info' do
    it 'resolves custom model' do
      Fastembed.register_model(
        model_name: 'custom/managed-model',
        dim: 768
      )

      info = Fastembed::ModelManagement.resolve_model_info('custom/managed-model')

      expect(info).to be_a(Fastembed::ModelInfo)
      expect(info.dim).to eq(768)
    end
  end
end
