# frozen_string_literal: true

require 'net/http'
require 'uri'
require 'json'
require 'fileutils'

module Fastembed
  # Handles model downloading and caching from HuggingFace
  #
  # Downloads ONNX models and tokenizer files from HuggingFace repositories,
  # caching them locally for subsequent use. Supports custom cache directories
  # via environment variables.
  #
  # @example Check cache location
  #   Fastembed::ModelManagement.cache_dir
  #   # => "/home/user/.cache/fastembed"
  #
  # @example Use custom cache directory
  #   Fastembed::ModelManagement.cache_dir = "/custom/path"
  #
  module ModelManagement
    # Base URL for HuggingFace API
    HF_API_BASE = 'https://huggingface.co'

    # Files required for model operation (in addition to model.onnx and tokenizer.json)
    REQUIRED_FILES = %w[
      config.json
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
    ].freeze

    class << self
      # Returns the cache directory for storing models
      #
      # Priority order:
      # 1. FASTEMBED_CACHE_PATH environment variable
      # 2. XDG_CACHE_HOME environment variable
      # 3. ~/.cache (fallback)
      #
      # @return [String] Absolute path to cache directory
      def cache_dir
        @cache_dir ||= begin
          base = ENV['FASTEMBED_CACHE_PATH'] ||
                 ENV['XDG_CACHE_HOME'] ||
                 File.join(Dir.home, '.cache')
          File.join(base, 'fastembed')
        end
      end

      # Set a custom cache directory
      # @!attribute [w] cache_dir
      # @return [String] Path to use as cache directory
      attr_writer :cache_dir

      # Returns the path to a cached model, downloading if necessary
      #
      # Downloads the model from HuggingFace if not already cached.
      # The model directory will contain the ONNX model file and tokenizer.
      #
      # @param model_name [String] Name of the model (e.g., "BAAI/bge-small-en-v1.5")
      # @param model_info [BaseModelInfo, nil] Optional pre-resolved model info
      # @param show_progress [Boolean] Whether to print download progress
      # @return [String] Absolute path to the model directory
      # @raise [DownloadError] If the download fails
      def retrieve_model(model_name, model_info: nil, show_progress: true)
        model_info ||= resolve_model_info(model_name)
        model_dir = model_directory(model_info)

        # Check if model is already cached
        return model_dir if model_cached?(model_dir, model_info)

        # Download model
        download_model(model_info, model_dir, show_progress: show_progress)
        model_dir
      end

      # Check if a model exists in cache
      #
      # @param model_dir [String] Path to model directory
      # @param model_info [BaseModelInfo] Model info with required file paths
      # @return [Boolean] True if model files exist
      def model_cached?(model_dir, model_info)
        return false unless Dir.exist?(model_dir)

        # Check for required files
        model_path = File.join(model_dir, model_info.model_file)
        tokenizer_path = File.join(model_dir, model_info.tokenizer_file)

        File.exist?(model_path) && File.exist?(tokenizer_path)
      end

      # Get the directory path for a model
      #
      # @param model_info [BaseModelInfo] Model info
      # @return [String] Path where model should be stored
      def model_directory(model_info)
        # Create a safe directory name from the model name
        safe_name = model_info.model_name.gsub('/', '--')
        File.join(cache_dir, 'models', safe_name)
      end

      # Resolve model name to ModelInfo from registry
      #
      # @param model_name [String] Model name to look up
      # @return [ModelInfo] The model information
      # @raise [ArgumentError] If model is not found in registry
      def resolve_model_info(model_name)
        model_info = SUPPORTED_MODELS[model_name]
        unless model_info
          raise ArgumentError,
                "Unknown model: #{model_name}. Use TextEmbedding.list_supported_models to see available models."
        end

        model_info
      end

      private

      def download_model(model_info, model_dir, show_progress: true)
        FileUtils.mkdir_p(model_dir)

        repo_id = model_info.hf_repo
        puts "Downloading model #{model_info.model_name} from #{repo_id}..." if show_progress

        # Download model file
        download_file(repo_id, model_info.model_file, model_dir, show_progress: show_progress)

        # Download tokenizer and config files
        files_to_download = REQUIRED_FILES + [model_info.tokenizer_file]
        files_to_download.uniq.each do |file|
          download_file(repo_id, file, model_dir, show_progress: show_progress,
                                                  required: file == model_info.tokenizer_file)
        end

        puts "Model downloaded successfully to #{model_dir}" if show_progress
      end

      def download_file(repo_id, file_path, model_dir, show_progress: true, required: true)
        # Determine the correct local path
        # If file_path contains directories (e.g., "onnx/model.onnx"), create them
        local_path = File.join(model_dir, file_path)
        FileUtils.mkdir_p(File.dirname(local_path))

        # Skip if already exists
        if File.exist?(local_path)
          puts "  #{file_path} (cached)" if show_progress
          return
        end

        url = "#{HF_API_BASE}/#{repo_id}/resolve/main/#{file_path}"
        puts "  Downloading #{file_path}..." if show_progress

        begin
          download_with_redirect(url, local_path, show_progress: show_progress)
        rescue StandardError => e
          raise DownloadError, "Failed to download #{file_path}: #{e.message}" if required

          puts "  #{file_path} (skipped - not available)" if show_progress
        end
      end

      def download_with_redirect(url, local_path, show_progress: true, max_redirects: 10)
        raise DownloadError, 'Too many redirects' if max_redirects <= 0

        uri = URI.parse(url)

        # Handle relative URLs by using https scheme
        unless uri.is_a?(URI::HTTP) || uri.is_a?(URI::HTTPS)
          # If it's a relative URL, we can't handle it
          raise DownloadError, "Invalid URL scheme: #{url}"
        end

        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https', read_timeout: 300,
                                            open_timeout: 30) do |http|
          request = Net::HTTP::Get.new(uri)
          request['User-Agent'] = "fastembed-ruby/#{VERSION}"

          response = http.request(request)

          case response
          when Net::HTTPSuccess
            File.binwrite(local_path, response.body)
          when Net::HTTPRedirection
            new_url = response['location']
            # Handle relative redirects
            new_url = "#{uri.scheme}://#{uri.host}#{new_url}" if new_url.start_with?('/')
            download_with_redirect(new_url, local_path, show_progress: show_progress, max_redirects: max_redirects - 1)
          else
            raise DownloadError, "HTTP #{response.code}: #{response.message}"
          end
        end
      end
    end
  end
end
