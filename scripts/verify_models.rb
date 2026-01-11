#!/usr/bin/env ruby
# frozen_string_literal: true

# Model Verification Script
# Downloads and tests all supported models to ensure they work correctly.
#
# Usage:
#   ruby scripts/verify_models.rb           # Run all model tests
#   ruby scripts/verify_models.rb --quick   # Quick test (first model of each type only)
#   ruby scripts/verify_models.rb --type embedding  # Test only embedding models

require 'bundler/setup'
require 'fastembed'
require 'optparse'

class ModelVerifier
  COLORS = {
    green: "\e[32m",
    red: "\e[31m",
    yellow: "\e[33m",
    cyan: "\e[36m",
    reset: "\e[0m"
  }.freeze

  def initialize(quick: false, type: nil)
    @quick = quick
    @type = type
    @results = { passed: [], failed: [], skipped: [] }
  end

  def run
    puts "#{COLORS[:cyan]}=== FastEmbed Model Verification ==#{COLORS[:reset]}"
    puts "Mode: #{@quick ? 'Quick' : 'Full'}"
    puts

    verify_embedding_models if should_test?(:embedding)
    verify_sparse_models if should_test?(:sparse)
    verify_late_interaction_models if should_test?(:late_interaction)
    verify_reranker_models if should_test?(:reranker)
    verify_image_models if should_test?(:image)

    print_summary
  end

  private

  def should_test?(model_type)
    @type.nil? || @type.to_sym == model_type
  end

  def verify_embedding_models
    section("Text Embedding Models")
    models = Fastembed::SUPPORTED_MODELS.keys
    models = [models.first] if @quick

    models.each do |model_name|
      verify_model(model_name, :embedding) do
        embedding = Fastembed::TextEmbedding.new(model_name: model_name, show_progress: false)
        vectors = embedding.embed(['Hello world', 'Test document']).to_a
        
        raise "Expected 2 vectors, got #{vectors.length}" unless vectors.length == 2
        raise "Vector dimension mismatch" unless vectors.first.length == embedding.dim
        
        "dim=#{embedding.dim}"
      end
    end
  end

  def verify_sparse_models
    section("Sparse Embedding Models")
    models = Fastembed::SUPPORTED_SPARSE_MODELS.keys
    models = [models.first] if @quick

    models.each do |model_name|
      verify_model(model_name, :sparse) do
        embedding = Fastembed::TextSparseEmbedding.new(model_name: model_name, show_progress: false)
        vectors = embedding.embed(['Hello world']).to_a
        
        raise "Expected sparse vector with indices" unless vectors.first[:indices].is_a?(Array)
        raise "Expected sparse vector with values" unless vectors.first[:values].is_a?(Array)
        
        "nnz=#{vectors.first[:indices].length}"
      end
    end
  end

  def verify_late_interaction_models
    section("Late Interaction Models")
    models = Fastembed::SUPPORTED_LATE_INTERACTION_MODELS.keys
    models = [models.first] if @quick

    models.each do |model_name|
      verify_model(model_name, :late_interaction) do
        embedding = Fastembed::LateInteractionTextEmbedding.new(model_name: model_name, show_progress: false)
        vectors = embedding.embed(['Hello world']).to_a
        
        raise "Expected token embeddings array" unless vectors.first.is_a?(Array)
        raise "Token embedding dimension mismatch" unless vectors.first.first.length == embedding.dim
        
        "dim=#{embedding.dim}, tokens=#{vectors.first.length}"
      end
    end
  end

  def verify_reranker_models
    section("Reranker Models")
    models = Fastembed::SUPPORTED_RERANKER_MODELS.keys
    models = [models.first] if @quick

    models.each do |model_name|
      verify_model(model_name, :reranker) do
        encoder = Fastembed::TextCrossEncoder.new(model_name: model_name, show_progress: false)
        results = encoder.rerank_with_scores(
          query: 'What is Ruby?',
          documents: ['Ruby is a programming language', 'The sky is blue']
        )
        
        raise "Expected ranked results" unless results.is_a?(Array)
        raise "Expected results with scores" unless results.first.key?(:score)
        
        "top_score=#{results.first[:score].round(4)}"
      end
    end
  end

  def verify_image_models
    section("Image Embedding Models")
    
    begin
      require 'mini_magick'
    rescue LoadError
      puts "#{COLORS[:yellow]}  Skipped (mini_magick not installed)#{COLORS[:reset]}"
      Fastembed::SUPPORTED_IMAGE_MODELS.keys.each do |model_name|
        @results[:skipped] << { name: model_name, type: :image, reason: 'mini_magick not installed' }
      end
      return
    end

    models = Fastembed::SUPPORTED_IMAGE_MODELS.keys
    models = [models.first] if @quick

    models.each do |model_name|
      verify_model(model_name, :image) do
        # Create a test image
        require 'tempfile'
        path = Tempfile.new(['test', '.png']).path
        MiniMagick::Tool::Convert.new do |convert|
          convert.size '224x224'
          convert.xc 'white'
          convert << path
        end

        begin
          embedding = Fastembed::ImageEmbedding.new(model_name: model_name, show_progress: false)
          vectors = embedding.embed(path).to_a
          
          raise "Expected image embedding" unless vectors.first.is_a?(Array)
          raise "Dimension mismatch" unless vectors.first.length == embedding.dim
          
          "dim=#{embedding.dim}"
        ensure
          File.delete(path) if File.exist?(path)
        end
      end
    end
  end

  def verify_model(model_name, type)
    print "  #{model_name}... "
    $stdout.flush
    
    start_time = Time.now
    begin
      result = yield
      elapsed = (Time.now - start_time).round(2)
      puts "#{COLORS[:green]}PASS#{COLORS[:reset]} (#{elapsed}s) [#{result}]"
      @results[:passed] << { name: model_name, type: type, time: elapsed }
    rescue => e
      elapsed = (Time.now - start_time).round(2)
      puts "#{COLORS[:red]}FAIL#{COLORS[:reset]} (#{elapsed}s)"
      puts "    Error: #{e.message}"
      @results[:failed] << { name: model_name, type: type, error: e.message }
    end
  end

  def section(title)
    puts "#{COLORS[:cyan]}#{title}:#{COLORS[:reset]}"
  end

  def print_summary
    puts
    puts "#{COLORS[:cyan]}=== Summary ==#{COLORS[:reset]}"
    puts "Passed: #{COLORS[:green]}#{@results[:passed].length}#{COLORS[:reset]}"
    puts "Failed: #{COLORS[:red]}#{@results[:failed].length}#{COLORS[:reset]}"
    puts "Skipped: #{COLORS[:yellow]}#{@results[:skipped].length}#{COLORS[:reset]}"
    puts

    unless @results[:failed].empty?
      puts "#{COLORS[:red]}Failed models:#{COLORS[:reset]}"
      @results[:failed].each do |r|
        puts "  - #{r[:name]} (#{r[:type]}): #{r[:error]}"
      end
    end

    exit(@results[:failed].empty? ? 0 : 1)
  end
end

# Parse options
options = { quick: false, type: nil }
OptionParser.new do |opts|
  opts.banner = "Usage: #{$PROGRAM_NAME} [options]"

  opts.on('-q', '--quick', 'Quick mode (first model of each type only)') do
    options[:quick] = true
  end

  opts.on('-t', '--type TYPE', %w[embedding sparse late_interaction reranker image],
          'Test only models of this type') do |type|
    options[:type] = type
  end

  opts.on('-h', '--help', 'Show this help') do
    puts opts
    exit
  end
end.parse!

ModelVerifier.new(**options).run
