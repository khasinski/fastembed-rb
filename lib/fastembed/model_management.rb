# frozen_string_literal: true

require "net/http"
require "uri"
require "json"
require "fileutils"

module Fastembed
  # Handles model downloading and caching
  module ModelManagement
    HF_API_BASE = "https://huggingface.co"
    REQUIRED_FILES = %w[
      config.json
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
    ].freeze

    class << self
      # Returns the cache directory for storing models
      # Priority: FASTEMBED_CACHE_PATH > XDG_CACHE_HOME > ~/.cache
      def cache_dir
        @cache_dir ||= begin
          base = ENV["FASTEMBED_CACHE_PATH"] ||
                 ENV["XDG_CACHE_HOME"] ||
                 File.join(Dir.home, ".cache")
          File.join(base, "fastembed")
        end
      end

      # Set a custom cache directory
      def cache_dir=(path)
        @cache_dir = path
      end

      # Returns the path to a cached model, downloading if necessary
      def retrieve_model(model_name, show_progress: true)
        model_info = resolve_model_info(model_name)
        model_dir = model_directory(model_info)

        # Check if model is already cached
        if model_cached?(model_dir, model_info)
          return model_dir
        end

        # Download model
        download_model(model_info, model_dir, show_progress: show_progress)
        model_dir
      end

      # Check if a model exists in cache
      def model_cached?(model_dir, model_info)
        return false unless Dir.exist?(model_dir)

        # Check for required files
        model_path = File.join(model_dir, model_info.model_file)
        tokenizer_path = File.join(model_dir, model_info.tokenizer_file)

        File.exist?(model_path) && File.exist?(tokenizer_path)
      end

      # Get the directory path for a model
      def model_directory(model_info)
        # Create a safe directory name from the model name
        safe_name = model_info.model_name.gsub("/", "--")
        File.join(cache_dir, "models", safe_name)
      end

      # Resolve model name to ModelInfo
      def resolve_model_info(model_name)
        model_info = SUPPORTED_MODELS[model_name]
        raise ArgumentError, "Unknown model: #{model_name}. Use TextEmbedding.list_supported_models to see available models." unless model_info

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
          download_file(repo_id, file, model_dir, show_progress: show_progress, required: file == model_info.tokenizer_file)
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
          if required
            raise DownloadError, "Failed to download #{file_path}: #{e.message}"
          else
            puts "  #{file_path} (skipped - not available)" if show_progress
          end
        end
      end

      def download_with_redirect(url, local_path, show_progress: true, max_redirects: 10)
        raise DownloadError, "Too many redirects" if max_redirects <= 0

        uri = URI.parse(url)

        # Handle relative URLs by using https scheme
        unless uri.is_a?(URI::HTTP) || uri.is_a?(URI::HTTPS)
          # If it's a relative URL, we can't handle it
          raise DownloadError, "Invalid URL scheme: #{url}"
        end

        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", read_timeout: 300, open_timeout: 30) do |http|
          request = Net::HTTP::Get.new(uri)
          request["User-Agent"] = "fastembed-ruby/#{VERSION}"

          response = http.request(request)

          case response
          when Net::HTTPSuccess
            File.open(local_path, "wb") do |file|
              file.write(response.body)
            end
          when Net::HTTPRedirection
            new_url = response["location"]
            # Handle relative redirects
            if new_url.start_with?("/")
              new_url = "#{uri.scheme}://#{uri.host}#{new_url}"
            end
            download_with_redirect(new_url, local_path, show_progress: show_progress, max_redirects: max_redirects - 1)
          else
            raise DownloadError, "HTTP #{response.code}: #{response.message}"
          end
        end
      end
    end
  end
end
