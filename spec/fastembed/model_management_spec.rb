# frozen_string_literal: true

require 'spec_helper'
require 'tmpdir'
require 'fileutils'
require 'webmock/rspec'

RSpec.describe Fastembed::ModelManagement do
  let(:tmp_cache_dir) { Dir.mktmpdir('fastembed-test') }

  before do
    described_class.cache_dir = tmp_cache_dir
  end

  after do
    FileUtils.rm_rf(tmp_cache_dir)
    described_class.cache_dir = nil
  end

  describe '.cache_dir' do
    context 'with FASTEMBED_CACHE_PATH env var' do
      it 'uses FASTEMBED_CACHE_PATH' do
        described_class.cache_dir = nil
        allow(ENV).to receive(:[]).and_call_original
        allow(ENV).to receive(:[]).with('FASTEMBED_CACHE_PATH').and_return('/custom/path')

        expect(described_class.cache_dir).to eq('/custom/path/fastembed')
      end
    end

    context 'with XDG_CACHE_HOME env var' do
      it 'uses XDG_CACHE_HOME when FASTEMBED_CACHE_PATH not set' do
        described_class.cache_dir = nil
        allow(ENV).to receive(:[]).and_call_original
        allow(ENV).to receive(:[]).with('FASTEMBED_CACHE_PATH').and_return(nil)
        allow(ENV).to receive(:[]).with('XDG_CACHE_HOME').and_return('/xdg/cache')

        expect(described_class.cache_dir).to eq('/xdg/cache/fastembed')
      end
    end
  end

  describe '.model_directory' do
    it 'creates a safe directory name from model name' do
      info = Fastembed::SUPPORTED_MODELS['BAAI/bge-small-en-v1.5']
      dir = described_class.model_directory(info)

      expect(dir).to include('BAAI--bge-small-en-v1.5')
      expect(dir).to include('models')
    end
  end

  describe '.model_cached?' do
    let(:model_info) { Fastembed::SUPPORTED_MODELS['BAAI/bge-small-en-v1.5'] }
    let(:model_dir) { described_class.model_directory(model_info) }

    it 'returns false when directory does not exist' do
      expect(described_class.model_cached?(model_dir, model_info)).to be false
    end

    it 'returns false when model file is missing' do
      FileUtils.mkdir_p(model_dir)
      FileUtils.touch(File.join(model_dir, model_info.tokenizer_file))

      expect(described_class.model_cached?(model_dir, model_info)).to be false
    end

    it 'returns false when tokenizer file is missing' do
      FileUtils.mkdir_p(File.join(model_dir, 'onnx'))
      FileUtils.touch(File.join(model_dir, model_info.model_file))

      expect(described_class.model_cached?(model_dir, model_info)).to be false
    end

    it 'returns true when both model and tokenizer exist' do
      FileUtils.mkdir_p(File.join(model_dir, 'onnx'))
      FileUtils.touch(File.join(model_dir, model_info.model_file))
      FileUtils.touch(File.join(model_dir, model_info.tokenizer_file))

      expect(described_class.model_cached?(model_dir, model_info)).to be true
    end
  end

  describe '.retrieve_model' do
    let(:model_info) { Fastembed::SUPPORTED_MODELS['BAAI/bge-small-en-v1.5'] }
    let(:model_dir) { described_class.model_directory(model_info) }

    context 'when model is cached' do
      before do
        FileUtils.mkdir_p(File.join(model_dir, 'onnx'))
        FileUtils.touch(File.join(model_dir, model_info.model_file))
        FileUtils.touch(File.join(model_dir, model_info.tokenizer_file))
      end

      it 'returns cached model directory without downloading' do
        # Should not make any HTTP requests
        result = described_class.retrieve_model('BAAI/bge-small-en-v1.5', show_progress: false)

        expect(result).to eq(model_dir)
      end
    end

    context 'when model is not cached' do
      before do
        WebMock.enable!
        # Stub all HuggingFace requests
        stub_request(:get, %r{huggingface.co/.+/resolve/main/.+})
          .to_return(status: 200, body: 'dummy content')
      end

      after do
        WebMock.disable!
      end

      it 'downloads model files' do
        result = described_class.retrieve_model('BAAI/bge-small-en-v1.5', show_progress: false)

        expect(result).to eq(model_dir)
        expect(Dir.exist?(model_dir)).to be true
      end

      it 'creates onnx subdirectory for model file' do
        described_class.retrieve_model('BAAI/bge-small-en-v1.5', show_progress: false)

        onnx_dir = File.join(model_dir, 'onnx')
        expect(Dir.exist?(onnx_dir)).to be true
      end
    end

    context 'when download fails' do
      before do
        WebMock.enable!
        stub_request(:get, %r{huggingface.co/.+/resolve/main/onnx/model.onnx})
          .to_return(status: 404, body: 'Not Found')
      end

      after do
        WebMock.disable!
      end

      it 'raises DownloadError for required files' do
        expect do
          described_class.retrieve_model('BAAI/bge-small-en-v1.5', show_progress: false)
        end.to raise_error(Fastembed::DownloadError, /Failed to download/)
      end
    end

    context 'with redirects' do
      before do
        WebMock.enable!
        stub_request(:get, %r{huggingface.co/.+/resolve/main/.+})
          .to_return(status: 302, headers: { 'Location' => 'https://cdn.example.com/file' })
        stub_request(:get, 'https://cdn.example.com/file')
          .to_return(status: 200, body: 'redirected content')
      end

      after do
        WebMock.disable!
      end

      it 'follows redirects' do
        result = described_class.retrieve_model('BAAI/bge-small-en-v1.5', show_progress: false)

        expect(result).to eq(model_dir)
      end
    end

    context 'with too many redirects' do
      before do
        WebMock.enable!
        stub_request(:get, %r{.+})
          .to_return(status: 302, headers: { 'Location' => 'https://example.com/redirect' })
      end

      after do
        WebMock.disable!
      end

      it 'raises DownloadError' do
        expect do
          described_class.retrieve_model('BAAI/bge-small-en-v1.5', show_progress: false)
        end.to raise_error(Fastembed::DownloadError, /Too many redirects/)
      end
    end
  end

  describe '.resolve_model_info' do
    it 'returns model info for valid model' do
      info = described_class.resolve_model_info('BAAI/bge-small-en-v1.5')

      expect(info).to be_a(Fastembed::ModelInfo)
      expect(info.dim).to eq(384)
    end

    it 'raises ArgumentError for unknown model' do
      expect do
        described_class.resolve_model_info('unknown/model')
      end.to raise_error(ArgumentError, /Unknown model/)
    end
  end
end
