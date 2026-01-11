# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Fastembed::ImageModelInfo do
  describe 'SUPPORTED_IMAGE_MODELS' do
    it 'contains the default image model' do
      expect(Fastembed::SUPPORTED_IMAGE_MODELS).to have_key(Fastembed::DEFAULT_IMAGE_MODEL)
    end

    it 'has HuggingFace source for all models' do
      Fastembed::SUPPORTED_IMAGE_MODELS.each do |name, info|
        expect(info.hf_repo).not_to be_nil, "Model #{name} missing HF repo"
      end
    end

    it 'has required attributes for all models' do
      Fastembed::SUPPORTED_IMAGE_MODELS.each_value do |info|
        expect(info.model_name).to be_a(String)
        expect(info.dim).to be_a(Integer)
        expect(info.description).to be_a(String)
        expect(info.size_in_gb).to be_a(Numeric)
        expect(info.image_size).to be_a(Integer)
        expect(info.mean).to be_an(Array)
        expect(info.std).to be_an(Array)
      end
    end

    it 'has valid image preprocessing parameters' do
      Fastembed::SUPPORTED_IMAGE_MODELS.each_value do |info|
        expect(info.image_size).to be > 0
        expect(info.mean.length).to eq(3) # RGB channels
        expect(info.std.length).to eq(3)
        info.mean.each { |v| expect(v).to be_between(0, 1) }
        info.std.each { |v| expect(v).to be > 0 }
      end
    end
  end

  describe '#initialize' do
    it 'creates image model info with all parameters' do
      info = described_class.new(
        model_name: 'test/image-model',
        dim: 512,
        description: 'Test image model',
        size_in_gb: 0.5,
        sources: { hf: 'test/image-model' },
        image_size: 224,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5]
      )

      expect(info.model_name).to eq('test/image-model')
      expect(info.dim).to eq(512)
      expect(info.image_size).to eq(224)
      expect(info.mean).to eq([0.5, 0.5, 0.5])
      expect(info.std).to eq([0.5, 0.5, 0.5])
    end

    it 'uses default preprocessing values' do
      info = described_class.new(
        model_name: 'test/model',
        dim: 512,
        description: 'Test',
        size_in_gb: 0.1,
        sources: {}
      )

      expect(info.image_size).to eq(224)
      expect(info.mean).to eq([0.485, 0.456, 0.406])
      expect(info.std).to eq([0.229, 0.224, 0.225])
    end
  end

  describe '#to_h' do
    it 'converts model info to hash' do
      info = Fastembed::SUPPORTED_IMAGE_MODELS['Qdrant/clip-ViT-B-32-vision']
      hash = info.to_h

      expect(hash[:model_name]).to eq('Qdrant/clip-ViT-B-32-vision')
      expect(hash[:dim]).to eq(512)
      expect(hash[:image_size]).to eq(224)
      expect(hash).to have_key(:description)
      expect(hash).to have_key(:sources)
    end
  end
end

RSpec.describe Fastembed::ImageEmbedding do
  describe '.list_supported_models' do
    it 'returns array of model hashes' do
      models = described_class.list_supported_models

      expect(models).to be_an(Array)
      expect(models).not_to be_empty
      expect(models.first).to have_key(:model_name)
      expect(models.first).to have_key(:dim)
      expect(models.first).to have_key(:image_size)
    end
  end

  describe '#initialize' do
    it 'raises error when mini_magick is not available' do
      # Hide mini_magick if it's loaded
      allow_any_instance_of(described_class).to receive(:require).with('mini_magick').and_raise(LoadError)

      expect do
        described_class.new
      end.to raise_error(Fastembed::Error, /mini_magick gem/)
    end
  end

  # Integration tests - only run if mini_magick and ImageMagick are available
  describe 'integration', :integration do
    before(:all) do
      @mini_magick_available = false
      @test_image_path = nil
      begin
        require 'mini_magick'
        # Test if ImageMagick is actually installed and working
        path = File.join(Dir.tmpdir, "fastembed_test_#{Process.pid}.png")
        MiniMagick::Tool::Magick.new do |magick|
          magick.size '224x224'
          magick << 'xc:white'
          magick << path
        end
        @test_image_path = path
        @mini_magick_available = true
      rescue LoadError, StandardError
        @mini_magick_available = false
      end
    end

    after(:all) do
      File.delete(@test_image_path) if @test_image_path && File.exist?(@test_image_path)
    end

    before do
      skip 'mini_magick not available' unless @mini_magick_available
    end

    let(:embedding) { described_class.new(show_progress: false) }
    let(:test_image_path) { @test_image_path }

    describe '#embed' do
      it 'generates embeddings for a single image' do
        vectors = embedding.embed(test_image_path).to_a

        expect(vectors.length).to eq(1)
        expect(vectors.first.length).to eq(512) # CLIP ViT-B/32 dim
        expect(vectors.first).to all(be_a(Float))
      end

      it 'returns normalized vectors' do
        vectors = embedding.embed(test_image_path).to_a
        vector = vectors.first

        norm = Math.sqrt(vector.sum { |v| v * v })
        expect(norm).to be_within(0.01).of(1.0)
      end

      it 'returns an Enumerator' do
        result = embedding.embed(test_image_path)
        expect(result).to be_an(Enumerator)
      end

      it 'handles empty array' do
        vectors = embedding.embed([]).to_a
        expect(vectors).to be_empty
      end
    end

    describe '#dim' do
      it 'returns the embedding dimension' do
        expect(embedding.dim).to eq(512)
      end
    end
  end
end
