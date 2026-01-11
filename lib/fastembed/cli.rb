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
        sparse_model: 'prithivida/Splade_PP_en_v1',
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
      when 'list-sparse'
        list_sparse
      when 'list-image'
        list_image
      when 'embed'
        embed
      when 'sparse-embed'
        sparse_embed
      when 'rerank'
        rerank
      when 'cache'
        cache_command
      when 'download'
        download_command
      when 'info'
        info_command
      when 'benchmark'
        benchmark_command
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
          embed           Generate dense embeddings for text
          sparse-embed    Generate sparse (SPLADE) embeddings for text
          rerank          Rerank documents by relevance to a query
          download        Pre-download a model for offline use
          info            Show detailed information about a model
          benchmark       Run performance benchmarks
          list-models     List available dense embedding models
          list-sparse     List available sparse embedding models
          list-rerankers  List available reranker models
          list-image      List available image embedding models
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

    def list_sparse
      puts 'Available sparse embedding models:'
      puts
      Fastembed::SUPPORTED_SPARSE_MODELS.each_value do |model|
        puts "  #{model.model_name}"
        puts "    Size: #{model.size_in_gb} GB"
        puts "    Description: #{model.description}"
        puts
      end
    end

    def list_image
      puts 'Available image embedding models:'
      puts
      Fastembed::SUPPORTED_IMAGE_MODELS.each_value do |model|
        puts "  #{model.model_name}"
        puts "    Dimensions: #{model.dim}"
        puts "    Image Size: #{model.image_size}x#{model.image_size}"
        puts "    Description: #{model.description}"
        puts
      end
    end

    def embed
      parse_embed_options
      texts = gather_texts

      if texts.empty?
        warn 'Error: No text provided. Pass text as arguments, use -i FILE, or pipe through stdin.'
        exit 1
      end

      show_progress = !@options[:quiet]
      embedding = Fastembed::TextEmbedding.new(model_name: @options[:model], show_progress: show_progress)

      embeddings = if @options[:show_progress] && !@options[:quiet]
                     embed_with_progress(embedding, texts)
                   else
                     embedding.embed(texts, batch_size: @options[:batch_size]).to_a
                   end

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

        opts.on('-i', '--input FILE', 'Read texts from file (one per line)') do |file|
          @options[:input_file] = file
        end

        opts.on('-q', '--quiet', 'Suppress progress output') do
          @options[:quiet] = true
        end

        opts.on('-p', '--progress', 'Show progress bar') do
          @options[:show_progress] = true
        end

        opts.on('-h', '--help', 'Show help') do
          puts opts
          exit 0
        end
      end.parse!(@argv)
    end

    def sparse_embed
      parse_sparse_embed_options
      texts = gather_texts

      if texts.empty?
        warn 'Error: No text provided. Pass text as arguments or pipe through stdin.'
        exit 1
      end

      embedding = Fastembed::TextSparseEmbedding.new(model_name: @options[:sparse_model])
      sparse_embeddings = embedding.embed(texts, batch_size: @options[:batch_size]).to_a

      output_sparse_embeddings(texts, sparse_embeddings)
    end

    def parse_sparse_embed_options
      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed sparse-embed [options] [text ...]'

        opts.on('-m', '--model MODEL', 'Model to use (default: prithivida/Splade_PP_en_v1)') do |m|
          @options[:sparse_model] = m
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

    def output_sparse_embeddings(texts, embeddings)
      case @options[:format]
      when 'json'
        output_sparse_json(texts, embeddings)
      when 'ndjson'
        output_sparse_ndjson(texts, embeddings)
      end
    end

    def output_sparse_json(texts, embeddings)
      result = texts.zip(embeddings).map do |text, emb|
        { text: text, indices: emb.indices, values: emb.values }
      end
      puts JSON.pretty_generate(result)
    end

    def output_sparse_ndjson(texts, embeddings)
      texts.zip(embeddings).each do |text, emb|
        puts JSON.generate({ text: text, indices: emb.indices, values: emb.values })
      end
    end

    def gather_texts
      # Priority: -i file > arguments > stdin
      if @options[:input_file]
        read_texts_from_file(@options[:input_file])
      elsif @argv.any?
        @argv
      elsif !$stdin.tty?
        $stdin.read.split("\n").reject(&:empty?)
      else
        []
      end
    end

    def read_texts_from_file(file_path)
      unless File.exist?(file_path)
        warn "Error: File not found: #{file_path}"
        exit 1
      end

      File.readlines(file_path, chomp: true).reject(&:empty?)
    end

    def embed_with_progress(embedding, texts)
      total = texts.length
      batch_size = @options[:batch_size]
      total_batches = (total.to_f / batch_size).ceil
      embeddings = embedding.embed(texts, batch_size: batch_size) do |progress|
        percent = (progress.current.to_f / progress.total * 100).round
        bar_width = 30
        filled = (bar_width * progress.current / progress.total).round
        bar = ('=' * filled) + ('-' * (bar_width - filled))
        $stderr.print "\r[#{bar}] #{percent}% (#{progress.current}/#{total_batches} batches)"
      end.map { |emb| emb }

      $stderr.puts
      embeddings
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

    # Download command - pre-download models for offline use
    def download_command
      parse_download_options

      model_name = @argv.shift
      if model_name.nil?
        warn 'Error: Model name required.'
        warn 'Usage: fastembed download <model-name>'
        warn "\nExamples:"
        warn '  fastembed download BAAI/bge-small-en-v1.5'
        warn '  fastembed download --type reranker cross-encoder/ms-marco-MiniLM-L-6-v2'
        exit 1
      end

      model_type = @options[:download_type] || :embedding

      begin
        model_info = resolve_model_for_download(model_name, model_type)
        puts "Downloading #{model_name}..."
        ModelManagement.retrieve_model(model_name, model_info: model_info, show_progress: true)
        puts 'Download complete!'
      rescue ArgumentError => e
        warn "Error: #{e.message}"
        exit 1
      rescue DownloadError => e
        warn "Download failed: #{e.message}"
        exit 1
      end
    end

    def parse_download_options
      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed download [options] <model-name>'

        opts.on('-t', '--type TYPE', %w[embedding reranker sparse late-interaction image],
                'Model type (embedding, reranker, sparse, late-interaction, image)') do |t|
          @options[:download_type] = t.tr('-', '_').to_sym
        end

        opts.on('-h', '--help', 'Show help') do
          puts opts
          puts "\nExamples:"
          puts '  fastembed download BAAI/bge-small-en-v1.5'
          puts '  fastembed download --type reranker cross-encoder/ms-marco-MiniLM-L-6-v2'
          exit 0
        end
      end.parse!(@argv)
    end

    def resolve_model_for_download(model_name, type)
      registry = case type
                 when :embedding
                   SUPPORTED_MODELS.merge(CustomModelRegistry.embedding_models)
                 when :reranker
                   SUPPORTED_RERANKER_MODELS.merge(CustomModelRegistry.reranker_models)
                 when :sparse
                   SUPPORTED_SPARSE_MODELS.merge(CustomModelRegistry.sparse_models)
                 when :late_interaction
                   SUPPORTED_LATE_INTERACTION_MODELS.merge(CustomModelRegistry.late_interaction_models)
                 when :image
                   SUPPORTED_IMAGE_MODELS
                 else
                   SUPPORTED_MODELS.merge(CustomModelRegistry.embedding_models)
                 end

      model_info = registry[model_name]
      raise ArgumentError, "Unknown #{type} model: #{model_name}" unless model_info

      model_info
    end

    # Info command - show detailed model information
    def info_command
      parse_info_options

      model_name = @argv.shift
      if model_name.nil?
        warn 'Error: Model name required.'
        warn 'Usage: fastembed info <model-name>'
        exit 1
      end

      model_info = find_model_info(model_name)
      if model_info.nil?
        warn "Unknown model: #{model_name}"
        exit 1
      end

      display_model_info(model_name, model_info)
    end

    def parse_info_options
      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed info <model-name>'

        opts.on('-h', '--help', 'Show help') do
          puts opts
          exit 0
        end
      end.parse!(@argv)
    end

    def find_model_info(model_name)
      # Search all registries
      SUPPORTED_MODELS[model_name] ||
        CustomModelRegistry.embedding_models[model_name] ||
        SUPPORTED_RERANKER_MODELS[model_name] ||
        CustomModelRegistry.reranker_models[model_name] ||
        SUPPORTED_SPARSE_MODELS[model_name] ||
        CustomModelRegistry.sparse_models[model_name] ||
        SUPPORTED_LATE_INTERACTION_MODELS[model_name] ||
        CustomModelRegistry.late_interaction_models[model_name] ||
        SUPPORTED_IMAGE_MODELS[model_name]
    end

    def display_model_info(model_name, info)
      puts "Model: #{model_name}"
      puts "  Description: #{info.description}"
      puts "  Size: #{info.size_in_gb} GB"
      puts "  Max Length: #{info.max_length} tokens"
      puts "  Model File: #{info.model_file}"
      puts "  Tokenizer: #{info.tokenizer_file}"

      # Type-specific info
      puts "  Dimensions: #{info.dim}" if info.respond_to?(:dim) && info.dim
      puts "  Pooling: #{info.pooling}" if info.respond_to?(:pooling)
      puts "  Normalize: #{info.normalize}" if info.respond_to?(:normalize)
      puts "  Image Size: #{info.image_size}x#{info.image_size}" if info.respond_to?(:image_size)

      # Source info
      puts "  HuggingFace: https://huggingface.co/#{info.sources[:hf]}" if info.sources[:hf]

      # Cache status
      cache_path = ModelManagement.model_directory(info)
      if ModelManagement.model_cached?(cache_path, info)
        size = directory_size(cache_path)
        puts "  Cached: Yes (#{format_size(size)})"
      else
        puts '  Cached: No'
      end
    end

    # Benchmark command - run performance benchmarks
    def benchmark_command
      parse_benchmark_options

      model_name = @options[:model]
      iterations = @options[:iterations]
      batch_size = @options[:batch_size]

      puts "Benchmarking #{model_name}..."
      puts "  Iterations: #{iterations}"
      puts "  Batch size: #{batch_size}"
      puts

      # Sample texts for benchmarking
      sample_texts = [
        'The quick brown fox jumps over the lazy dog.',
        'Machine learning is transforming how we build software.',
        'Ruby is a dynamic, open source programming language.',
        'Embeddings convert text into numerical vectors.'
      ] * ((batch_size / 4) + 1)
      sample_texts = sample_texts.first(batch_size)

      begin
        # Load model (measure load time)
        puts 'Loading model...'
        load_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        embedding = Fastembed::TextEmbedding.new(model_name: model_name)
        load_time = Process.clock_gettime(Process::CLOCK_MONOTONIC) - load_start
        puts "  Load time: #{format('%.2f', load_time)}s"
        puts

        # Warmup
        puts 'Warming up...'
        embedding.embed(sample_texts.first(4)).to_a

        # Benchmark
        puts "Running #{iterations} iterations..."
        times = []
        iterations.times do |i|
          start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          embedding.embed(sample_texts, batch_size: batch_size).to_a
          elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
          times << elapsed
          print "\r  Progress: #{i + 1}/#{iterations}"
        end
        puts

        # Results
        puts
        puts 'Results:'
        avg_time = times.sum / times.length
        min_time = times.min
        max_time = times.max
        throughput = batch_size / avg_time

        puts "  Avg time: #{format('%.3f', avg_time)}s"
        puts "  Min time: #{format('%.3f', min_time)}s"
        puts "  Max time: #{format('%.3f', max_time)}s"
        puts "  Throughput: #{format('%.1f', throughput)} texts/sec"
        puts "  Dimensions: #{embedding.dim}"
      rescue StandardError => e
        warn "Benchmark failed: #{e.message}"
        exit 1
      end
    end

    def parse_benchmark_options
      @options[:iterations] = 10

      OptionParser.new do |opts|
        opts.banner = 'Usage: fastembed benchmark [options]'

        opts.on('-m', '--model MODEL', 'Model to benchmark (default: BAAI/bge-small-en-v1.5)') do |m|
          @options[:model] = m
        end

        opts.on('-n', '--iterations N', Integer, 'Number of iterations (default: 10)') do |n|
          @options[:iterations] = n
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
  end
end
