# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::Validators do
  describe '.validate_documents!' do
    it 'returns array unchanged when valid' do
      docs = %w[hello world]
      result = described_class.validate_documents!(docs)

      expect(result).to eq(docs)
    end

    it 'wraps single string in array' do
      result = described_class.validate_documents!('hello')

      expect(result).to eq(['hello'])
    end

    it 'raises error for nil input' do
      expect do
        described_class.validate_documents!(nil)
      end.to raise_error(ArgumentError, 'documents cannot be nil')
    end

    it 'raises error for nil element in array' do
      expect do
        described_class.validate_documents!(['hello', nil, 'world'])
      end.to raise_error(ArgumentError, 'document at index 1 cannot be nil')
    end

    it 'raises error for nil at first position' do
      expect do
        described_class.validate_documents!([nil, 'hello'])
      end.to raise_error(ArgumentError, 'document at index 0 cannot be nil')
    end

    it 'allows empty array' do
      result = described_class.validate_documents!([])

      expect(result).to eq([])
    end

    it 'allows empty strings' do
      result = described_class.validate_documents!(['', 'hello', ''])

      expect(result).to eq(['', 'hello', ''])
    end
  end

  describe '.validate_rerank_input!' do
    it 'returns documents unchanged when valid' do
      docs = %w[doc1 doc2]
      result = described_class.validate_rerank_input!(query: 'test', documents: docs)

      expect(result).to eq(docs)
    end

    it 'raises error for nil query' do
      expect do
        described_class.validate_rerank_input!(query: nil, documents: ['doc'])
      end.to raise_error(ArgumentError, 'query cannot be nil')
    end

    it 'raises error for nil documents' do
      expect do
        described_class.validate_rerank_input!(query: 'test', documents: nil)
      end.to raise_error(ArgumentError, 'documents cannot be nil')
    end

    it 'raises error for nil element in documents' do
      expect do
        described_class.validate_rerank_input!(query: 'test', documents: ['doc1', nil])
      end.to raise_error(ArgumentError, 'document at index 1 cannot be nil')
    end

    it 'allows empty documents array' do
      result = described_class.validate_rerank_input!(query: 'test', documents: [])

      expect(result).to eq([])
    end

    it 'allows empty query string' do
      result = described_class.validate_rerank_input!(query: '', documents: ['doc'])

      expect(result).to eq(['doc'])
    end
  end
end
