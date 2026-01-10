#!/usr/bin/env ruby
# frozen_string_literal: true

require 'bundler/setup'
require 'fastembed'
require 'benchmark'

# Sample texts of varying lengths
SHORT_TEXTS = [
  'Hello world',
  'Ruby is great',
  'Machine learning',
  'Vector embeddings',
  'Semantic search'
].freeze

MEDIUM_TEXTS = [
  'The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing tests.',
  'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
  'Ruby on Rails is a server-side web application framework written in Ruby under the MIT License.',
  'Vector databases store embeddings and enable fast similarity search across millions of documents.',
  'Natural language processing helps computers understand, interpret, and generate human language.'
].freeze

LONG_TEXTS = [
  'Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving.',
  'Ruby is an interpreted, high-level, general-purpose programming language which supports multiple programming paradigms. It was designed with an emphasis on programming productivity and simplicity. In Ruby, everything is an object, including primitive data types. It was developed in the mid-1990s by Yukihiro Matsumoto in Japan. Ruby is dynamically typed and uses garbage collection and just-in-time compilation.',
  'Text embeddings are dense vector representations of text that capture semantic meaning. They are produced by machine learning models trained on large corpora of text data. These embeddings enable semantic similarity calculations, clustering, and information retrieval tasks. Modern embedding models like BERT, Sentence Transformers, and OpenAI embeddings have revolutionized natural language processing applications.',
  'Vector databases are specialized database systems designed to store and query high-dimensional vector data efficiently. They use approximate nearest neighbor algorithms like HNSW, IVF, and PQ to enable fast similarity search at scale. Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and pgvector. They are essential infrastructure for semantic search, recommendation systems, and RAG applications.',
  'The Transformer architecture, introduced in the paper "Attention Is All You Need", revolutionized natural language processing. It relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. This enables much more parallelization and has led to significant improvements in translation quality. Transformers are the foundation of modern language models like GPT, BERT, and T5.'
].freeze

def print_separator
  puts '-' * 70
end

def format_rate(count, time)
  rate = count / time
  "#{rate.round(1)} docs/sec"
end

def profile_batch(embedding, texts, batch_size, iterations = 3)
  times = []
  iterations.times do
    GC.start
    time = Benchmark.realtime do
      embedding.embed(texts, batch_size: batch_size).to_a
    end
    times << time
  end
  times.min # Return best time
end

puts '=' * 70
puts 'FASTEMBED-RB PERFORMANCE PROFILE'
puts '=' * 70
puts
puts "Ruby version: #{RUBY_VERSION}"
puts "Platform: #{RUBY_PLATFORM}"
puts "Fastembed version: #{Fastembed::VERSION}"
puts

# Model loading benchmark
print_separator
puts 'MODEL LOADING TIME'
print_separator

models = [
  'BAAI/bge-small-en-v1.5',
  'sentence-transformers/all-MiniLM-L6-v2'
]

models.each do |model_name|
  # Ensure model is downloaded first
  Fastembed::TextEmbedding.new(model_name: model_name)
  GC.start

  time = Benchmark.realtime do
    Fastembed::TextEmbedding.new(model_name: model_name)
  end
  puts "#{model_name}: #{(time * 1000).round(1)}ms"
end

puts

# Single document latency
print_separator
puts 'SINGLE DOCUMENT LATENCY (lower is better)'
print_separator

embedding = Fastembed::TextEmbedding.new
warmup = embedding.embed(['warmup']).to_a # Warm up

[SHORT_TEXTS.first, MEDIUM_TEXTS.first, LONG_TEXTS.first].each_with_index do |text, i|
  label = %w[Short Medium Long][i]
  times = []
  10.times do
    time = Benchmark.realtime { embedding.embed([text]).to_a }
    times << time
  end
  avg = times.sum / times.length
  min = times.min
  puts "#{label} text (#{text.length} chars): avg #{(avg * 1000).round(2)}ms, min #{(min * 1000).round(2)}ms"
end

puts

# Throughput benchmarks
print_separator
puts 'THROUGHPUT (higher is better)'
print_separator

[10, 100, 500, 1000].each do |count|
  texts = MEDIUM_TEXTS.cycle.take(count)

  [32, 64, 128, 256].each do |batch_size|
    next if batch_size > count

    time = profile_batch(embedding, texts, batch_size)
    rate = format_rate(count, time)
    puts "#{count} docs, batch #{batch_size}: #{rate} (#{(time * 1000).round(1)}ms total)"
  end
  puts
end

# Memory efficiency test
print_separator
puts 'LAZY EVALUATION TEST'
print_separator

texts = MEDIUM_TEXTS.cycle.take(1000)
processed = 0

time = Benchmark.realtime do
  embedding.embed(texts, batch_size: 64).each do |_vec|
    processed += 1
    break if processed >= 100 # Only process first 100
  end
end

puts "Processed #{processed}/1000 documents in #{(time * 1000).round(1)}ms"
puts '(Lazy evaluation means we only computed embeddings for documents we needed)'

puts

# Embedding quality sanity check
print_separator
puts 'EMBEDDING QUALITY SANITY CHECK'
print_separator

test_pairs = [
  ['dog', 'puppy', 'high'],
  ['dog', 'cat', 'medium'],
  ['dog', 'airplane', 'low'],
  ['machine learning', 'artificial intelligence', 'high'],
  ['machine learning', 'cooking recipes', 'low']
]

def cosine_similarity(a, b)
  a.zip(b).sum { |x, y| x * y }
end

test_pairs.each do |text1, text2, expected|
  vecs = embedding.embed([text1, text2]).to_a
  sim = cosine_similarity(vecs[0], vecs[1])
  status = case expected
           when 'high' then sim > 0.7 ? 'PASS' : 'FAIL'
           when 'medium' then sim > 0.4 && sim < 0.8 ? 'PASS' : 'FAIL'
           when 'low' then sim < 0.5 ? 'PASS' : 'FAIL'
           end
  puts "#{status}: '#{text1}' vs '#{text2}' = #{sim.round(3)} (expected #{expected})"
end

puts

# Compare with batch sizes
print_separator
puts 'OPTIMAL BATCH SIZE ANALYSIS'
print_separator

texts = MEDIUM_TEXTS.cycle.take(500)
results = {}

[1, 8, 16, 32, 64, 128, 256, 512].each do |batch_size|
  time = profile_batch(embedding, texts, batch_size, 2)
  rate = 500.0 / time
  results[batch_size] = rate
  puts "Batch #{batch_size.to_s.rjust(3)}: #{rate.round(1)} docs/sec"
end

optimal = results.max_by { |_, v| v }
puts
puts "Optimal batch size: #{optimal[0]} (#{optimal[1].round(1)} docs/sec)"

puts
puts '=' * 70
puts 'PROFILE COMPLETE'
puts '=' * 70
