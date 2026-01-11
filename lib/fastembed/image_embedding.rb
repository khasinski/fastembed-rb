# frozen_string_literal: true

module Fastembed
  # Model information for image embedding models
  class ImageModelInfo
    include BaseModelInfo

    attr_reader :dim, :image_size, :mean, :std

    def initialize(
      model_name:,
      dim:,
      description:,
      size_in_gb:,
      sources:,
      model_file: 'model.onnx',
      image_size: 224,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225]
    )
      initialize_base(
        model_name: model_name,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        tokenizer_file: '',
        max_length: 0
      )
      @dim = dim
      @image_size = image_size
      @mean = mean
      @std = std
    end

    def to_h
      {
        model_name: model_name,
        dim: dim,
        description: description,
        size_in_gb: size_in_gb,
        sources: sources,
        model_file: model_file,
        image_size: image_size
      }
    end
  end

  # Registry of supported image embedding models
  SUPPORTED_IMAGE_MODELS = {
    'Qdrant/clip-ViT-B-32-vision' => ImageModelInfo.new(
      model_name: 'Qdrant/clip-ViT-B-32-vision',
      dim: 512,
      description: 'CLIP ViT-B/32 vision encoder',
      size_in_gb: 0.34,
      sources: { hf: 'Qdrant/clip-ViT-B-32-vision' },
      model_file: 'model.onnx',
      image_size: 224
    ),
    'Qdrant/resnet50-onnx' => ImageModelInfo.new(
      model_name: 'Qdrant/resnet50-onnx',
      dim: 2048,
      description: 'ResNet-50 image encoder',
      size_in_gb: 0.10,
      sources: { hf: 'Qdrant/resnet50-onnx' },
      model_file: 'model.onnx',
      image_size: 224
    ),
    'jinaai/jina-clip-v1' => ImageModelInfo.new(
      model_name: 'jinaai/jina-clip-v1',
      dim: 768,
      description: 'Jina CLIP v1 vision encoder',
      size_in_gb: 0.35,
      sources: { hf: 'jinaai/jina-clip-v1' },
      model_file: 'onnx/vision_model.onnx',
      image_size: 224
    )
  }.freeze

  DEFAULT_IMAGE_MODEL = 'Qdrant/clip-ViT-B-32-vision'

  # Image embedding model for converting images to vectors
  #
  # Supports CLIP and ResNet models for image search and multimodal applications.
  # Requires the mini_magick gem for image processing.
  #
  # @example Basic usage
  #   image_embed = Fastembed::ImageEmbedding.new
  #   vectors = image_embed.embed(["path/to/image.jpg"]).to_a
  #
  # @example With URLs
  #   vectors = image_embed.embed(["https://example.com/image.jpg"]).to_a
  #
  class ImageEmbedding
    attr_reader :model_name, :model_info, :dim

    # Initialize an image embedding model
    #
    # @param model_name [String] Name of the model to use
    # @param cache_dir [String, nil] Custom cache directory for models
    # @param threads [Integer, nil] Number of threads for ONNX Runtime
    # @param providers [Array<String>, nil] ONNX execution providers
    # @param show_progress [Boolean] Whether to show download progress
    def initialize(
      model_name: DEFAULT_IMAGE_MODEL,
      cache_dir: nil,
      threads: nil,
      providers: nil,
      show_progress: true
    )
      require_mini_magick!

      @model_name = model_name
      @threads = threads
      @providers = providers || ['CPUExecutionProvider']

      ModelManagement.cache_dir = cache_dir if cache_dir

      @model_info = resolve_model_info(model_name)
      @model_dir = retrieve_model(model_name, show_progress: show_progress)
      @dim = @model_info.dim

      setup_model
    end

    # Generate embeddings for images
    #
    # @param images [Array<String>, String] Image path(s) or URL(s) to embed
    # @param batch_size [Integer] Number of images to process at once
    # @yield [Progress] Optional progress callback called after each batch
    # @return [Enumerator] Lazy enumerator yielding embedding vectors
    def embed(images, batch_size: 32, &progress_callback)
      images = [images] if images.is_a?(String)
      return Enumerator.new { |_| } if images.empty?

      total_batches = (images.length.to_f / batch_size).ceil

      Enumerator.new do |yielder|
        images.each_slice(batch_size).with_index(1) do |batch, batch_num|
          embeddings = compute_embeddings(batch)
          embeddings.each { |emb| yielder << emb }

          if progress_callback
            progress = Progress.new(current: batch_num, total: total_batches, batch_size: batch_size)
            progress_callback.call(progress)
          end
        end
      end
    end

    # Generate embeddings asynchronously
    #
    # @param images [Array<String>, String] Image path(s) or URL(s) to embed
    # @param batch_size [Integer] Number of images to process at once
    # @return [Async::Future] Future that resolves to array of embedding vectors
    def embed_async(images, batch_size: 32)
      Async::Future.new { embed(images, batch_size: batch_size).to_a }
    end

    # List all supported image models
    #
    # @return [Array<Hash>] Array of model information hashes
    def self.list_supported_models
      SUPPORTED_IMAGE_MODELS.values.map(&:to_h)
    end

    private

    def require_mini_magick!
      require 'mini_magick'
    rescue LoadError
      raise Error, 'Image embedding requires the mini_magick gem. Add it to your Gemfile: gem "mini_magick"'
    end

    def resolve_model_info(model_name)
      info = SUPPORTED_IMAGE_MODELS[model_name]
      raise Error, "Unknown image model: #{model_name}" unless info

      info
    end

    def retrieve_model(model_name, show_progress:)
      ModelManagement.retrieve_model(
        model_name,
        model_info: @model_info,
        show_progress: show_progress
      )
    end

    def setup_model
      model_path = File.join(@model_dir, @model_info.model_file)
      raise Error, "Model file not found: #{model_path}" unless File.exist?(model_path)

      options = {}
      options[:inter_op_num_threads] = @threads if @threads
      options[:intra_op_num_threads] = @threads if @threads

      @session = OnnxRuntime::InferenceSession.new(
        model_path,
        **options,
        providers: @providers
      )
    end

    def compute_embeddings(image_paths)
      # Preprocess images into tensor
      tensors = image_paths.map { |path| preprocess_image(path) }

      # Stack into batch [batch, channels, height, width]
      batch_tensor = tensors

      # Run inference
      input_name = @session.inputs.first[:name]
      outputs = @session.run(nil, { input_name => batch_tensor })

      # Extract and normalize embeddings
      embeddings = outputs.first
      embeddings.map { |emb| normalize_embedding(emb) }
    end

    def preprocess_image(image_path)
      # Load image
      image = load_image(image_path)

      # Resize to model's expected size
      size = @model_info.image_size
      image.resize "#{size}x#{size}!"

      # Convert to RGB tensor and normalize
      pixels = extract_pixels(image)
      normalize_pixels(pixels)
    end

    def load_image(path)
      raise Error, "Image file not found: #{path}" if !path.start_with?('http://', 'https://') && !File.exist?(path)

      MiniMagick::Image.open(path)
    end

    def extract_pixels(image)
      # Get raw RGB pixel data using ImageMagick's export
      # depth:8 ensures 8-bit per channel, and 'RGB' gives us raw RGB bytes
      pixels_str = image.run_command('convert', image.path, '-depth', '8', 'RGB:-')

      # Convert to array of RGB values [0-255]
      pixels_str.unpack('C*')
    end

    def normalize_pixels(pixels)
      size = @model_info.image_size
      mean = @model_info.mean
      std = @model_info.std

      # Convert from [H, W, C] flat array to [C, H, W] tensor
      channels = 3
      tensor = Array.new(channels) { Array.new(size) { Array.new(size) } }

      pixels.each_with_index do |pixel, i|
        h = (i / 3) / size
        w = (i / 3) % size
        c = i % 3

        # Normalize: (pixel/255 - mean) / std
        normalized = ((pixel / 255.0) - mean[c]) / std[c]
        tensor[c][h][w] = normalized
      end

      tensor
    end

    def normalize_embedding(embedding)
      # L2 normalize the embedding
      embedding = embedding.flatten if embedding.is_a?(Array) && embedding.first.is_a?(Array)
      norm = Math.sqrt(embedding.sum { |x| x * x })
      return embedding if norm.zero?

      embedding.map { |x| x / norm }
    end
  end
end
