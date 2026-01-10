#!/usr/bin/env ruby
# frozen_string_literal: true

# Benchmark script for TextCrossEncoder (reranker) performance

require 'bundler/setup'
require 'fastembed'
require 'benchmark'

QUERY = 'What is machine learning?'

DOCUMENTS = [
  'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
  'Ruby on Rails is a server-side web application framework written in Ruby under the MIT License.',
  'Deep learning uses neural networks with many layers to model complex patterns in data.',
  'Vector databases store embeddings and enable fast similarity search across millions of documents.',
  'Supervised learning requires labeled training data to learn the mapping from inputs to outputs.',
  'Natural language processing helps computers understand, interpret, and generate human language.',
  'Random forests are ensemble learning methods that construct multiple decision trees.',
  'The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing tests.',
  'Gradient descent is an optimization algorithm used to minimize the loss function in ML models.',
  'Transformers use self-attention mechanisms to process sequential data in parallel.'
].freeze

def print_separator
  puts '-' * 70
end

puts '=' * 70
puts 'RERANKER (CROSS-ENCODER) PERFORMANCE BENCHMARK'
puts '=' * 70
puts
puts "Ruby version: #{RUBY_VERSION}"
puts "Fastembed version: #{Fastembed::VERSION}"
puts

# Model loading benchmark
print_separator
puts 'MODEL LOADING TIME'
print_separator

Fastembed::SUPPORTED_RERANKER_MODELS.each_key do |model_name|
  # Ensure model is downloaded first
  begin
    Fastembed::TextCrossEncoder.new(model_name: model_name)
  rescue StandardError
    puts "#{model_name}: (skipped - not available)"
    next
  end
  GC.start

  time = Benchmark.realtime do
    Fastembed::TextCrossEncoder.new(model_name: model_name)
  end
  puts "#{model_name}: #{(time * 1000).round(1)}ms"
end

puts

# Use default model for latency tests
reranker = Fastembed::TextCrossEncoder.new

# Single query latency
print_separator
puts 'SINGLE QUERY LATENCY (reranking against 10 documents)'
print_separator

times = []
20.times do
  GC.start
  time = Benchmark.realtime do
    reranker.rerank(query: QUERY, documents: DOCUMENTS)
  end
  times << time
end

avg = times.sum / times.length
min = times.min
max = times.max
puts "Average: #{(avg * 1000).round(2)}ms"
puts "Min: #{(min * 1000).round(2)}ms"
puts "Max: #{(max * 1000).round(2)}ms"
puts

# Throughput with varying document counts
print_separator
puts 'THROUGHPUT VS DOCUMENT COUNT'
print_separator

[10, 50, 100, 200].each do |doc_count|
  docs = DOCUMENTS.cycle.take(doc_count)

  times = []
  3.times do
    GC.start
    time = Benchmark.realtime do
      reranker.rerank(query: QUERY, documents: docs, batch_size: 64)
    end
    times << time
  end

  min_time = times.min
  rate = doc_count / min_time
  puts "#{doc_count} documents: #{rate.round(1)} docs/sec (#{(min_time * 1000).round(1)}ms)"
end

puts

# Batch size optimization
print_separator
puts 'BATCH SIZE OPTIMIZATION (100 documents)'
print_separator

docs = DOCUMENTS.cycle.take(100)
results = {}

[8, 16, 32, 64, 128].each do |batch_size|
  times = []
  3.times do
    GC.start
    time = Benchmark.realtime do
      reranker.rerank(query: QUERY, documents: docs, batch_size: batch_size)
    end
    times << time
  end

  min_time = times.min
  rate = 100.0 / min_time
  results[batch_size] = rate
  puts "Batch #{batch_size.to_s.rjust(3)}: #{rate.round(1)} docs/sec"
end

optimal = results.max_by { |_, v| v }
puts
puts "Optimal batch size: #{optimal[0]} (#{optimal[1].round(1)} docs/sec)"

puts

# Quality check
print_separator
puts 'RERANKING QUALITY CHECK'
print_separator

results = reranker.rerank_with_scores(query: QUERY, documents: DOCUMENTS, top_k: 5)

puts "Query: '#{QUERY}'"
puts
puts 'Top 5 results:'
results.each_with_index do |result, i|
  score = result[:score]
  doc = result[:document][0, 60]
  puts "#{i + 1}. (#{score.round(3)}) #{doc}..."
end

puts
puts '=' * 70
puts 'BENCHMARK COMPLETE'
puts '=' * 70
