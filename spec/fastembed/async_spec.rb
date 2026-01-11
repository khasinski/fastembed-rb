# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::Async::Future do
  describe '#initialize' do
    it 'executes block in background thread' do
      executed = false
      future = described_class.new { executed = true }
      future.wait
      expect(executed).to be true
    end
  end

  describe '#complete?' do
    it 'returns false while running' do
      future = described_class.new { sleep 0.1 }
      expect(future.complete?).to be false
    end

    it 'returns true after completion' do
      future = described_class.new { 'result' }
      future.wait
      expect(future.complete?).to be true
    end
  end

  describe '#pending?' do
    it 'returns true while running' do
      future = described_class.new { sleep 0.1 }
      expect(future.pending?).to be true
    end

    it 'returns false after completion' do
      future = described_class.new { 'result' }
      future.wait
      expect(future.pending?).to be false
    end
  end

  describe '#success?' do
    it 'returns true on successful completion' do
      future = described_class.new { 'result' }
      future.wait
      expect(future.success?).to be true
    end

    it 'returns false on error' do
      future = described_class.new { raise 'error' }
      future.wait
      expect(future.success?).to be false
    end
  end

  describe '#failure?' do
    it 'returns true on error' do
      future = described_class.new { raise 'error' }
      future.wait
      expect(future.failure?).to be true
    end

    it 'returns false on success' do
      future = described_class.new { 'result' }
      future.wait
      expect(future.failure?).to be false
    end
  end

  describe '#value' do
    it 'returns the result' do
      future = described_class.new { 42 }
      expect(future.value).to eq(42)
    end

    it 'blocks until complete' do
      start = Time.now
      future = described_class.new { sleep 0.05; 'done' }
      result = future.value
      elapsed = Time.now - start

      expect(result).to eq('done')
      expect(elapsed).to be >= 0.04
    end

    it 'raises error if operation failed' do
      future = described_class.new { raise ArgumentError, 'test error' }
      expect { future.value }.to raise_error(ArgumentError, 'test error')
    end

    it 'supports timeout' do
      future = described_class.new { sleep 1 }
      expect(future.wait(timeout: 0.01)).to be false
    end
  end

  describe '#error' do
    it 'returns nil on success' do
      future = described_class.new { 'result' }
      future.wait
      expect(future.error).to be_nil
    end

    it 'returns the error on failure' do
      future = described_class.new { raise 'my error' }
      future.wait
      expect(future.error).to be_a(RuntimeError)
      expect(future.error.message).to eq('my error')
    end
  end

  describe '#then' do
    it 'transforms the result' do
      future = described_class.new { 10 }
      transformed = future.then { |v| v * 2 }
      expect(transformed.value).to eq(20)
    end
  end

  describe '#rescue' do
    it 'handles errors' do
      future = described_class.new { raise 'error' }
      rescued = future.rescue { |_e| 'recovered' }
      expect(rescued.value).to eq('recovered')
    end

    it 'passes through successful results' do
      future = described_class.new { 'success' }
      rescued = future.rescue { |_e| 'recovered' }
      expect(rescued.value).to eq('success')
    end
  end
end

RSpec.describe Fastembed::Async do
  describe '.all' do
    it 'waits for all futures and returns results' do
      futures = [
        Fastembed::Async::Future.new { 1 },
        Fastembed::Async::Future.new { 2 },
        Fastembed::Async::Future.new { 3 }
      ]
      results = described_class.all(futures)
      expect(results).to eq([1, 2, 3])
    end
  end

  describe '.race' do
    it 'returns the first completed result' do
      futures = [
        Fastembed::Async::Future.new { sleep 0.1; 'slow' },
        Fastembed::Async::Future.new { 'fast' }
      ]
      result = described_class.race(futures)
      expect(result).to eq('fast')
    end
  end
end

RSpec.describe 'TextEmbedding async methods', :integration do
  let(:embedding) { Fastembed::TextEmbedding.new(show_progress: false) }

  describe '#embed_async' do
    it 'returns a Future' do
      future = embedding.embed_async(['hello'])
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces correct embeddings' do
      future = embedding.embed_async(['hello', 'world'])
      vectors = future.value

      expect(vectors.length).to eq(2)
      expect(vectors[0].length).to eq(embedding.dim)
    end

    it 'allows parallel processing' do
      batch1 = Array.new(10) { 'document one' }
      batch2 = Array.new(10) { 'document two' }

      futures = [
        embedding.embed_async(batch1),
        embedding.embed_async(batch2)
      ]

      results = futures.map(&:value)
      expect(results[0].length).to eq(10)
      expect(results[1].length).to eq(10)
    end
  end

  describe '#query_embed_async' do
    it 'returns a Future' do
      future = embedding.query_embed_async('what is ruby?')
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces embeddings' do
      future = embedding.query_embed_async(['query1', 'query2'])
      vectors = future.value
      expect(vectors.length).to eq(2)
    end
  end

  describe '#passage_embed_async' do
    it 'returns a Future' do
      future = embedding.passage_embed_async('ruby is a programming language')
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces embeddings' do
      future = embedding.passage_embed_async(['passage1', 'passage2'])
      vectors = future.value
      expect(vectors.length).to eq(2)
    end
  end
end

RSpec.describe 'SparseEmbedding async methods', :integration do
  before(:all) do
    begin
      @embedding = Fastembed::TextSparseEmbedding.new(show_progress: false)
      @sparse_available = true
    rescue StandardError => e
      @sparse_available = false
      @skip_reason = e.message
    end
  end

  before do
    skip @skip_reason unless @sparse_available
  end

  let(:embedding) { @embedding }

  describe '#embed_async' do
    it 'returns a Future' do
      future = embedding.embed_async(['hello'])
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces sparse embeddings' do
      future = embedding.embed_async(['hello', 'world'])
      vectors = future.value

      expect(vectors.length).to eq(2)
      expect(vectors[0]).to be_a(Fastembed::SparseEmbedding)
      expect(vectors[0].indices).to be_an(Array)
      expect(vectors[0].values).to be_an(Array)
    end
  end

  describe '#query_embed_async' do
    it 'returns a Future' do
      future = embedding.query_embed_async('what is ruby?')
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces sparse embeddings' do
      future = embedding.query_embed_async(['query1', 'query2'])
      vectors = future.value
      expect(vectors.length).to eq(2)
      expect(vectors[0]).to be_a(Fastembed::SparseEmbedding)
    end
  end

  describe '#passage_embed_async' do
    it 'returns a Future' do
      future = embedding.passage_embed_async('ruby is a programming language')
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces sparse embeddings' do
      future = embedding.passage_embed_async(['passage1', 'passage2'])
      vectors = future.value
      expect(vectors.length).to eq(2)
      expect(vectors[0]).to be_a(Fastembed::SparseEmbedding)
    end
  end
end

RSpec.describe 'LateInteractionEmbedding async methods', :integration do
  before(:all) do
    begin
      @embedding = Fastembed::LateInteractionTextEmbedding.new(show_progress: false)
      @late_interaction_available = true
    rescue StandardError => e
      @late_interaction_available = false
      @skip_reason = e.message
    end
  end

  before do
    skip @skip_reason unless @late_interaction_available
  end

  let(:embedding) { @embedding }

  describe '#embed_async' do
    it 'returns a Future' do
      future = embedding.embed_async(['hello'])
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces token embeddings' do
      future = embedding.embed_async(['hello', 'world'])
      vectors = future.value

      expect(vectors.length).to eq(2)
      # Each document has a LateInteractionEmbedding with multiple token embeddings
      expect(vectors[0]).to be_a(Fastembed::LateInteractionEmbedding)
      expect(vectors[0].to_a.first.length).to eq(embedding.dim)
    end
  end

  describe '#query_embed_async' do
    it 'returns a Future' do
      future = embedding.query_embed_async('what is ruby?')
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces token embeddings' do
      future = embedding.query_embed_async(['query1', 'query2'])
      vectors = future.value
      expect(vectors.length).to eq(2)
      expect(vectors[0]).to be_a(Fastembed::LateInteractionEmbedding)
    end
  end

  describe '#passage_embed_async' do
    it 'returns a Future' do
      future = embedding.passage_embed_async('ruby is a programming language')
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces token embeddings' do
      future = embedding.passage_embed_async(['passage1', 'passage2'])
      vectors = future.value
      expect(vectors.length).to eq(2)
      expect(vectors[0]).to be_a(Fastembed::LateInteractionEmbedding)
    end
  end
end

RSpec.describe 'TextCrossEncoder async methods', :integration do
  before(:all) do
    begin
      @encoder = Fastembed::TextCrossEncoder.new(show_progress: false)
      @reranker_available = true
    rescue StandardError => e
      @reranker_available = false
      @skip_reason = e.message
    end
  end

  before do
    skip @skip_reason unless @reranker_available
  end

  let(:encoder) { @encoder }

  describe '#rerank_async' do
    it 'returns a Future' do
      future = encoder.rerank_async(query: 'test', documents: %w[doc1 doc2])
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces scores' do
      future = encoder.rerank_async(query: 'Ruby', documents: ['Ruby code', 'Python code'])
      scores = future.value

      expect(scores.length).to eq(2)
      expect(scores[0]).to be_a(Float)
    end
  end

  describe '#rerank_with_scores_async' do
    it 'returns a Future' do
      future = encoder.rerank_with_scores_async(query: 'test', documents: %w[doc1 doc2])
      expect(future).to be_a(Fastembed::Async::Future)
    end

    it 'produces sorted results with scores' do
      future = encoder.rerank_with_scores_async(
        query: 'Ruby programming',
        documents: ['Ruby code', 'Python code', 'JavaScript code']
      )
      results = future.value

      expect(results.length).to eq(3)
      expect(results[0]).to have_key(:document)
      expect(results[0]).to have_key(:score)
      expect(results[0]).to have_key(:index)
      # Results should be sorted by score (highest first)
      scores = results.map { |r| r[:score] }
      expect(scores).to eq(scores.sort.reverse)
    end

    it 'respects top_k parameter' do
      future = encoder.rerank_with_scores_async(
        query: 'test',
        documents: %w[a b c d e],
        top_k: 2
      )
      results = future.value

      expect(results.length).to eq(2)
    end
  end
end

RSpec.describe 'Concurrent async operations across model types', :integration do
  before(:all) do
    begin
      @text_embedding = Fastembed::TextEmbedding.new(show_progress: false)
      @text_available = true
    rescue StandardError => e
      @text_available = false
      @skip_reason = e.message
    end
  end

  before do
    skip @skip_reason unless @text_available
  end

  it 'supports running multiple text embedding futures concurrently' do
    futures = [
      @text_embedding.embed_async(['hello world']),
      @text_embedding.embed_async(['another document'])
    ]

    results = futures.map(&:value)

    expect(results[0].first).to be_an(Array)
    expect(results[0].first.length).to eq(@text_embedding.dim)
    expect(results[1].first.length).to eq(@text_embedding.dim)
  end
end
