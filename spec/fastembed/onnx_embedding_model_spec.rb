# frozen_string_literal: true

require 'spec_helper'
require 'tmpdir'
require 'fileutils'

RSpec.describe Fastembed::OnnxEmbeddingModel do
  describe 'initialization' do
    let(:model_info) { Fastembed::SUPPORTED_MODELS['BAAI/bge-small-en-v1.5'] }
    let(:model_dir) { Fastembed::ModelManagement.retrieve_model('BAAI/bge-small-en-v1.5') }

    it 'loads model and tokenizer successfully' do
      model = described_class.new(model_info, model_dir)
      expect(model).to be_a(described_class)
      expect(model.model_info).to eq(model_info)
      expect(model.model_dir).to eq(model_dir)
    end

    it 'raises error if model file is missing' do
      expect do
        described_class.new(model_info, '/nonexistent/path')
      end.to raise_error(Fastembed::Error, /Model file not found/)
    end

    it 'raises error if tokenizer file is missing' do
      # Create a temp dir with model but no tokenizer
      Dir.mktmpdir do |tmpdir|
        onnx_dir = File.join(tmpdir, 'onnx')
        FileUtils.mkdir_p(onnx_dir)
        FileUtils.cp(File.join(model_dir, model_info.model_file), onnx_dir)

        expect do
          described_class.new(model_info, tmpdir)
        end.to raise_error(Fastembed::Error, /Tokenizer file not found/)
      end
    end

    it 'accepts threads parameter' do
      model = described_class.new(model_info, model_dir, threads: 2)
      expect(model).to be_a(described_class)
    end

    it 'accepts providers parameter' do
      model = described_class.new(model_info, model_dir, providers: ['CPUExecutionProvider'])
      expect(model).to be_a(described_class)
    end
  end

  describe '#embed', :integration do
    let(:model_info) { Fastembed::SUPPORTED_MODELS['BAAI/bge-small-en-v1.5'] }
    let(:model_dir) { Fastembed::ModelManagement.retrieve_model('BAAI/bge-small-en-v1.5') }
    let(:model) { described_class.new(model_info, model_dir) }

    it 'embeds single text' do
      embeddings = model.embed(['Hello world'])
      expect(embeddings.length).to eq(1)
      expect(embeddings.first.length).to eq(384)
    end

    it 'embeds multiple texts' do
      embeddings = model.embed(%w[Hello World])
      expect(embeddings.length).to eq(2)
      expect(embeddings).to all(have_attributes(length: 384))
    end

    it 'returns normalized vectors' do
      embeddings = model.embed(['Test'])
      norm = Math.sqrt(embeddings.first.sum { |v| v * v })
      expect(norm).to be_within(0.01).of(1.0)
    end

    it 'handles empty string' do
      embeddings = model.embed([''])
      expect(embeddings.length).to eq(1)
      expect(embeddings.first.length).to eq(384)
    end

    it 'handles unicode text' do
      embeddings = model.embed(['Hello 世界 🌍'])
      expect(embeddings.length).to eq(1)
      expect(embeddings.first.length).to eq(384)
    end

    it 'truncates long text' do
      long_text = 'word ' * 1000
      embeddings = model.embed([long_text])
      expect(embeddings.length).to eq(1)
      expect(embeddings.first.length).to eq(384)
    end

    it 'produces consistent results' do
      text = 'Consistency test'
      result1 = model.embed([text])
      result2 = model.embed([text])
      expect(result1).to eq(result2)
    end

    it 'produces different embeddings for different texts' do
      embeddings = model.embed(%w[Cats Programming])
      expect(embeddings[0]).not_to eq(embeddings[1])
    end
  end
end
