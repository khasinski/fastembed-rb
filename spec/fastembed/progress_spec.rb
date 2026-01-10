# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::Progress do
  describe '#initialize' do
    it 'stores progress information' do
      progress = described_class.new(current: 5, total: 10, batch_size: 32)

      expect(progress.current).to eq(5)
      expect(progress.total).to eq(10)
      expect(progress.batch_size).to eq(32)
    end
  end

  describe '#percentage' do
    it 'returns completion percentage as float' do
      progress = described_class.new(current: 5, total: 10, batch_size: 32)

      expect(progress.percentage).to eq(0.5)
    end

    it 'returns 1.0 when complete' do
      progress = described_class.new(current: 10, total: 10, batch_size: 32)

      expect(progress.percentage).to eq(1.0)
    end

    it 'returns 1.0 for zero total' do
      progress = described_class.new(current: 0, total: 0, batch_size: 32)

      expect(progress.percentage).to eq(1.0)
    end
  end

  describe '#percent' do
    it 'returns integer percentage' do
      progress = described_class.new(current: 1, total: 3, batch_size: 32)

      expect(progress.percent).to eq(33)
    end
  end

  describe '#documents_processed' do
    it 'returns number of documents processed' do
      progress = described_class.new(current: 3, total: 10, batch_size: 32)

      expect(progress.documents_processed).to eq(96)
    end
  end

  describe '#complete?' do
    it 'returns true when complete' do
      progress = described_class.new(current: 10, total: 10, batch_size: 32)

      expect(progress.complete?).to be true
    end

    it 'returns false when not complete' do
      progress = described_class.new(current: 5, total: 10, batch_size: 32)

      expect(progress.complete?).to be false
    end
  end

  describe '#to_s' do
    it 'returns readable string' do
      progress = described_class.new(current: 5, total: 10, batch_size: 32)

      expect(progress.to_s).to eq('Progress(5/10, 50%)')
    end
  end
end

RSpec.describe 'TextEmbedding with progress callback' do
  let(:embedding) { Fastembed::TextEmbedding.new }

  it 'calls progress callback for each batch' do
    documents = Array.new(100) { 'test document' }
    progress_calls = []

    embedding.embed(documents, batch_size: 32) do |progress|
      progress_calls << progress
    end.to_a

    expect(progress_calls.length).to eq(4) # 100 docs / 32 batch = 4 batches (ceil)
    expect(progress_calls.first.current).to eq(1)
    expect(progress_calls.last.current).to eq(4)
    expect(progress_calls.last.complete?).to be true
  end

  it 'works without progress callback' do
    documents = %w[hello world]

    result = embedding.embed(documents).to_a

    expect(result.length).to eq(2)
  end
end
