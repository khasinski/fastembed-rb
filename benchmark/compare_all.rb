#!/usr/bin/env ruby
# frozen_string_literal: true

# Unified benchmark comparing Ruby fastembed with Python fastembed
# Runs both implementations and reports side-by-side results

require 'bundler/setup'
require 'fastembed'
require 'benchmark'
require 'json'
require 'open3'

TEXTS = [
  'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
  'Ruby on Rails is a server-side web application framework written in Ruby under the MIT License.',
  'Vector databases store embeddings and enable fast similarity search across millions of documents.',
  'Natural language processing helps computers understand, interpret, and generate human language.',
  'The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing tests.'
].freeze

def run_ruby_benchmark
  puts 'Running Ruby benchmark...'
  results = {}

  # Model loading
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  embedding = Fastembed::TextEmbedding.new(model_name: 'BAAI/bge-small-en-v1.5', show_progress: false)
  results[:load_time] = ((Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) * 1000).round(1)

  # Warmup
  embedding.embed(['warmup']).to_a

  # Single document latency
  times = []
  10.times do
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    embedding.embed([TEXTS.first]).to_a
    times << (Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) * 1000
  end
  results[:single_latency] = times.min.round(2)

  # Throughput tests
  [100, 500, 1000].each do |count|
    texts = TEXTS.cycle.take(count)
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    embedding.embed(texts, batch_size: 64).to_a
    elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
    results[:"throughput_#{count}"] = (count / elapsed).round(1)
  end

  results
end

def run_python_benchmark
  puts 'Running Python benchmark...'

  python_script = <<~PYTHON
    import json
    import time
    from fastembed import TextEmbedding

    TEXTS = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Ruby on Rails is a server-side web application framework written in Ruby under the MIT License.",
        "Vector databases store embeddings and enable fast similarity search across millions of documents.",
        "Natural language processing helps computers understand, interpret, and generate human language.",
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typing tests."
    ]

    results = {}

    # Model loading
    start = time.time()
    embedding = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    results["load_time"] = round((time.time() - start) * 1000, 1)

    # Warmup
    list(embedding.embed(["warmup"]))

    # Single document latency
    times = []
    for _ in range(10):
        start = time.time()
        list(embedding.embed([TEXTS[0]]))
        times.append((time.time() - start) * 1000)
    results["single_latency"] = round(min(times), 2)

    # Throughput tests
    for count in [100, 500, 1000]:
        texts = (TEXTS * (count // len(TEXTS) + 1))[:count]
        start = time.time()
        list(embedding.embed(texts, batch_size=64))
        elapsed = time.time() - start
        results[f"throughput_{count}"] = round(count / elapsed, 1)

    print(json.dumps(results))
  PYTHON

  stdout, status = Open3.capture2('python3', '-c', python_script)

  if status.success?
    JSON.parse(stdout)
  else
    puts 'Warning: Python benchmark failed. Is fastembed installed? (pip install fastembed)'
    nil
  end
rescue Errno::ENOENT
  puts 'Warning: Python not found'
  nil
end

def print_comparison(ruby_results, python_results)
  puts
  puts '=' * 70
  puts 'RUBY vs PYTHON FASTEMBED COMPARISON'
  puts '=' * 70
  puts

  metrics = [
    [:load_time, 'Model load time', 'ms', :lower_better],
    [:single_latency, 'Single doc latency', 'ms', :lower_better],
    [:throughput_100, '100 docs throughput', 'docs/sec', :higher_better],
    [:throughput_500, '500 docs throughput', 'docs/sec', :higher_better],
    [:throughput_1000, '1000 docs throughput', 'docs/sec', :higher_better]
  ]

  puts format('%-25s %15s %15s %10s', 'Metric', 'Ruby', 'Python', 'Winner')
  puts '-' * 70

  metrics.each do |key, label, unit, direction|
    ruby_val = ruby_results[key]
    python_val = python_results&.fetch(key.to_s, nil)

    if python_val
      if direction == :lower_better
        winner = ruby_val < python_val ? 'Ruby' : 'Python'
        ratio = python_val / ruby_val
      else
        winner = ruby_val > python_val ? 'Ruby' : 'Python'
        ratio = ruby_val / python_val
      end
      ratio_str = winner == 'Ruby' ? "(#{ratio.round(1)}x)" : ''
      winner_str = "#{winner} #{ratio_str}"
    else
      winner_str = 'N/A'
    end

    ruby_str = "#{ruby_val} #{unit}"
    python_str = python_val ? "#{python_val} #{unit}" : 'N/A'

    puts format('%-25s %15s %15s %10s', label, ruby_str, python_str, winner_str)
  end

  puts
end

# Run benchmarks
ruby_results = run_ruby_benchmark
python_results = run_python_benchmark

print_comparison(ruby_results, python_results)

puts 'Summary:'
puts '- Both use the same ONNX Runtime and HuggingFace Tokenizers'
puts '- Performance differences come from language overhead and batching'
puts '- Ruby tends to win on latency, Python on large batch throughput'
puts
