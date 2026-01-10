# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::TextCrossEncoder do
  describe '.list_supported_models' do
    it 'returns array of model hashes' do
      models = described_class.list_supported_models
      expect(models).to be_an(Array)
      expect(models).not_to be_empty
    end

    it 'contains the default model' do
      models = described_class.list_supported_models
      model_names = models.map { |m| m[:model_name] }
      expect(model_names).to include(Fastembed::DEFAULT_RERANKER_MODEL)
    end
  end

  describe 'initialization' do
    it 'raises error for unknown model' do
      expect do
        described_class.new(model_name: 'unknown/model')
      end.to raise_error(Fastembed::Error, /Unknown reranker model/)
    end
  end

  context 'with loaded model', :integration do
    subject(:reranker) { described_class.new }

    describe '#rerank' do
      it 'returns scores for documents' do
        scores = reranker.rerank(
          query: 'What is machine learning?',
          documents: ['ML is AI', 'The sky is blue']
        )

        expect(scores).to be_an(Array)
        expect(scores.length).to eq(2)
        expect(scores).to all(be_a(Float))
      end

      it 'returns empty array for empty documents' do
        scores = reranker.rerank(query: 'test', documents: [])
        expect(scores).to eq([])
      end

      it 'scores relevant documents higher than irrelevant ones' do
        scores = reranker.rerank(
          query: 'What is the capital of France?',
          documents: [
            'Paris is the capital of France.',
            'The best pizza is in Italy.',
            'France is a country in Europe.'
          ]
        )

        # Paris answer should score highest
        expect(scores[0]).to be > scores[1]
        expect(scores[0]).to be > scores[2]
      end

      it 'handles batch processing' do
        documents = Array.new(100) { |i| "Document number #{i} with some content." }
        scores = reranker.rerank(
          query: 'Find document 50',
          documents: documents,
          batch_size: 32
        )

        expect(scores.length).to eq(100)
      end
    end

    describe '#rerank_with_scores' do
      let(:results) do
        reranker.rerank_with_scores(
          query: 'What is Ruby?',
          documents: [
            'Python is a programming language.',
            'Ruby is a dynamic programming language.',
            'The weather is nice.'
          ]
        )
      end

      it 'returns an array of results' do
        expect(results).to be_an(Array)
        expect(results.length).to eq(3)
      end

      it 'includes document, score, and index keys' do
        expect(results.first).to include(:document, :score, :index)
      end

      it 'sorts results by score descending' do
        scores = results.map { |r| r[:score] }
        expect(scores).to eq(scores.sort.reverse)
      end

      it 'ranks relevant documents first' do
        expect(results.first[:document]).to include('Ruby')
      end

      it 'respects top_k parameter' do
        results = reranker.rerank_with_scores(
          query: 'test query',
          documents: %w[doc1 doc2 doc3 doc4 doc5],
          top_k: 2
        )

        expect(results.length).to eq(2)
      end

      it 'returns all results when top_k is nil' do
        results = reranker.rerank_with_scores(
          query: 'test',
          documents: %w[a b c],
          top_k: nil
        )

        expect(results.length).to eq(3)
      end
    end
  end
end
