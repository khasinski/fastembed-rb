#!/usr/bin/env ruby
# frozen_string_literal: true

require 'bundler/setup'
require 'fastembed'

def memory_mb
  `ps -o rss= -p #{Process.pid}`.to_i / 1024.0
end

def print_memory(label)
  puts "#{label}: #{memory_mb.round(1)} MB"
end

puts '=' * 60
puts 'MEMORY PROFILING'
puts '=' * 60
puts

print_memory('Initial')

# Load model
embedding = Fastembed::TextEmbedding.new
print_memory('After model load')

# Generate sample texts
texts = Array.new(1000) { |i| "This is document number #{i} with some content for embedding." }

# Process in batches
GC.start
print_memory('Before embedding 1000 docs')

vectors = embedding.embed(texts, batch_size: 64).to_a
print_memory('After embedding 1000 docs (holding results)')

# Clear vectors
vectors = nil
GC.start
sleep 0.1
print_memory('After clearing vectors + GC')

# Test lazy evaluation memory efficiency
puts
puts 'Testing lazy evaluation memory efficiency...'
print_memory('Before lazy processing')

count = 0
embedding.embed(texts, batch_size: 64).each do |_vec|
  count += 1
  # Don't store vectors, just count them
end
puts "Processed #{count} vectors without storing"

GC.start
sleep 0.1
print_memory('After lazy processing + GC')

# Stress test - multiple rounds
puts
puts 'Stress test - 5 rounds of 1000 docs each...'
5.times do |round|
  embedding.embed(texts, batch_size: 64).to_a
  GC.start
  print_memory("After round #{round + 1}")
end

puts
puts '=' * 60
puts 'MEMORY PROFILE COMPLETE'
puts '=' * 60
