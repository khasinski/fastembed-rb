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
      @options = { format: 'json', model: 'BAAI/bge-small-en-v1.5', batch_size: 256 }
    end

    def run
      command = parse_global_options

      case command
      when 'list-models', 'models'
        list_models
      when 'embed'
        embed
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
          embed         Generate embeddings for text
          list-models   List available embedding models
          version       Show version
          help          Show this help message

        Run 'fastembed <command> --help' for command-specific options.
      HELP
    end

    def list_models
      puts 'Available models:'
      puts
      Fastembed::SUPPORTED_MODELS.each_value do |model|
        puts "  #{model.model_name}"
        puts "    Dimensions: #{model.dim}"
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
  end
end
