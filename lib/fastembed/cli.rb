# frozen_string_literal: true

require 'optparse'
require 'json'
require_relative '../fastembed'

module Fastembed
  # Command-line interface for fastembed
  class CLI
    FORMATS = %w[json ndjson csv].freeze

    def initialize(argv)
      @argv = argv
      @options = {
        format: 'json',
        model: 'BAAI/bge-small-en-v1.5',
        reranker_model: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        batch_size: 256,
        top_k: nil
      }
    end

    def run
      command = parse_global_options

      case command
      when 'list-models', 'models'
        list_models
      when 'list-rerankers', 'rerankers'
        list_rerankers
      when 'embed'
        embed
      when 'rerank'
        rerank
      when 'cache'
        cache_command
      when 'version', '-v', '--version'
        puts "fastembed #{Fastembed::VERSION}"
      when 'help', nil
        puts global_help
      else
        warn "Unknown command: #{command}"
        warn global_help
        exit 1
      end
    end

    private

    def parse_global_options
      return @argv.shift if @argv.empty? || !@argv.first.start_with?('-')

      global_parser.order!(@argv)
      @argv.shift
    end

    def global_parser
      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed [options] <command> [command-options]'
        opts.on('-v', '--version', 'Show version') do
          puts "fastembed #{Fastembed::VERSION}"
          exit 0
        end
        opts.on('-h', '--help', 'Show help') do
          puts global_help
          exit 0
        end
      end
    end

    def global_help
      <<~HELP
        fastembed - Fast text embeddings for Ruby

        Usage: fastembed <command> [options]

        Commands:
          embed           Generate embeddings for text
          rerank          Rerank documents by relevance to a query
          list-models     List available embedding models
          list-rerankers  List available reranker models
          cache           Manage model cache (clear, info)
          version         Show version
          help            Show this help message

        Run 'fastembed <command> --help' for command-specific options.
      HELP
    end

    def list_models
      puts 'Available embedding models:'
      puts
      Fastembed::SUPPORTED_MODELS.each_value do |model|
        puts "  #{model.model_name}"
        puts "    Dimensions: #{model.dim}"
        puts "    Description: #{model.description}"
        puts
      end
    end

    def list_rerankers
      puts 'Available reranker models:'
      puts
      Fastembed::SUPPORTED_RERANKER_MODELS.each_value do |model|
        puts "  #{model.model_name}"
        puts "    Size: #{model.size_in_gb} GB"
        puts "    Description: #{model.description}"
        puts
      end
    end

    def embed
      parse_embed_options
      texts = gather_texts

      if texts.empty?
        warn 'Error: No text provided. Pass text as arguments or pipe through stdin.'
        exit 1
      end

      embedding = Fastembed::TextEmbedding.new(model_name: @options[:model])
      embeddings = embedding.embed(texts, batch_size: @options[:batch_size]).to_a

      output_embeddings(texts, embeddings)
    end

    def parse_embed_options
      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed embed [options] [text ...]'

        opts.on('-m', '--model MODEL', 'Model to use (default: BAAI/bge-small-en-v1.5)') do |m|
          @options[:model] = m
        end

        opts.on('-f', '--format FORMAT', FORMATS, "Output format: #{FORMATS.join(', ')} (default: json)") do |f|
          @options[:format] = f
        end

        opts.on('-b', '--batch-size SIZE', Integer, 'Batch size (default: 256)') do |b|
          @options[:batch_size] = b
        end

        opts.on('-h', '--help', 'Show help') do
          puts opts
          exit 0
        end
      end.parse!(@argv)
    end

    def gather_texts
      if @argv.any?
        @argv
      elsif !$stdin.tty?
        $stdin.read.split("\n").reject(&:empty?)
      else
        []
      end
    end

    def output_embeddings(texts, embeddings)
      case @options[:format]
      when 'json'
        output_json(texts, embeddings)
      when 'ndjson'
        output_ndjson(texts, embeddings)
      when 'csv'
        output_csv(embeddings)
      end
    end

    def output_json(texts, embeddings)
      result = texts.zip(embeddings).map do |text, embedding|
        { text: text, embedding: embedding }
      end
      puts JSON.pretty_generate(result)
    end

    def output_ndjson(texts, embeddings)
      texts.zip(embeddings).each do |text, embedding|
        puts JSON.generate({ text: text, embedding: embedding })
      end
    end

    def output_csv(embeddings)
      embeddings.each do |embedding|
        puts embedding.join(',')
      end
    end

    def rerank
      parse_rerank_options

      if @options[:query].nil? || @options[:query].empty?
        warn 'Error: Query is required. Use -q or --query to specify.'
        exit 1
      end

      documents = gather_texts
      if documents.empty?
        warn 'Error: No documents provided. Pass documents as arguments or pipe through stdin.'
        exit 1
      end

      reranker = Fastembed::TextCrossEncoder.new(model_name: @options[:reranker_model])
      results = reranker.rerank_with_scores(
        query: @options[:query],
        documents: documents,
        top_k: @options[:top_k],
        batch_size: @options[:batch_size]
      )

      output_rerank_results(results)
    end

    def parse_rerank_options
      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed rerank -q QUERY [options] [documents ...]'

        opts.on('-q', '--query QUERY', 'Query to rank documents against (required)') do |q|
          @options[:query] = q
        end

        opts.on('-m', '--model MODEL', 'Reranker model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)') do |m|
          @options[:reranker_model] = m
        end

        opts.on('-k', '--top-k K', Integer, 'Return only top K results') do |k|
          @options[:top_k] = k
        end

        opts.on('-f', '--format FORMAT', %w[json ndjson], 'Output format: json, ndjson (default: json)') do |f|
          @options[:format] = f
        end

        opts.on('-b', '--batch-size SIZE', Integer, 'Batch size (default: 256)') do |b|
          @options[:batch_size] = b
        end

        opts.on('-h', '--help', 'Show help') do
          puts opts
          exit 0
        end
      end.parse!(@argv)
    end

    def output_rerank_results(results)
      case @options[:format]
      when 'json'
        puts JSON.pretty_generate(results)
      when 'ndjson'
        results.each { |r| puts JSON.generate(r) }
      end
    end

    def cache_command
      subcommand = @argv.shift

      case subcommand
      when 'clear'
        cache_clear
      when 'info'
        cache_info
      when 'help', nil, '--help', '-h'
        puts cache_help
      else
        warn "Unknown cache subcommand: #{subcommand}"
        warn cache_help
        exit 1
      end
    end

    def cache_help
      <<~HELP
        Usage: fastembed cache <subcommand>

        Subcommands:
          clear   Remove all cached models
          info    Show cache directory and size

        Examples:
          fastembed cache info
          fastembed cache clear
      HELP
    end

    def cache_clear
      cache_path = ModelManagement.cache_dir
      models_path = File.join(cache_path, 'models')

      unless Dir.exist?(models_path)
        puts 'Cache is empty.'
        return
      end

      # Count models before clearing
      model_count = Dir.glob(File.join(models_path, '*')).count { |f| File.directory?(f) }

      if model_count.zero?
        puts 'Cache is empty.'
        return
      end

      print "Remove #{model_count} cached model(s)? [y/N] "
      response = $stdin.gets&.strip&.downcase

      if response == 'y'
        FileUtils.rm_rf(models_path)
        puts 'Cache cleared.'
      else
        puts 'Aborted.'
      end
    end

    def cache_info
      cache_path = ModelManagement.cache_dir
      models_path = File.join(cache_path, 'models')

      puts "Cache directory: #{cache_path}"
      puts

      unless Dir.exist?(models_path)
        puts 'No models cached.'
        return
      end

      models = Dir.glob(File.join(models_path, '*')).select { |f| File.directory?(f) }

      if models.empty?
        puts 'No models cached.'
        return
      end

      total_size = 0
      puts 'Cached models:'
      models.each do |model_dir|
        name = File.basename(model_dir).gsub('--', '/')
        size = directory_size(model_dir)
        total_size += size
        puts "  #{name} (#{format_size(size)})"
      end

      puts
      puts "Total: #{models.count} model(s), #{format_size(total_size)}"
    end

    def directory_size(path)
      Dir.glob(File.join(path, '**', '*'))
         .select { |f| File.file?(f) }
         .sum { |f| File.size(f) }
    end

    def format_size(bytes)
      if bytes >= 1_073_741_824
        format('%.2f GB', bytes / 1_073_741_824.0)
      elsif bytes >= 1_048_576
        format('%.2f MB', bytes / 1_048_576.0)
      elsif bytes >= 1024
        format('%.2f KB', bytes / 1024.0)
      else
        "#{bytes} B"
      end
    end
  end
end
