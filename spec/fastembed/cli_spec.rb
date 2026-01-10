# frozen_string_literal: true

require 'spec_helper'
require 'fastembed/cli'
require 'json'
require 'open3'

RSpec.describe Fastembed::CLI do
  let(:exe_path) { File.expand_path('../../exe/fastembed', __dir__) }

  describe 'help command' do
    it 'shows help message' do
      cli = described_class.new(['help'])
      expect { cli.run }.to output(/Usage: fastembed/).to_stdout
    end

    it 'shows help with --help flag' do
      cli = described_class.new(['--help'])
      expect { cli.run }.to output(/Usage: fastembed/).to_stdout.and raise_error(SystemExit)
    end
  end

  describe 'version command' do
    it 'shows version' do
      cli = described_class.new(['version'])
      expect { cli.run }.to output(/fastembed #{Fastembed::VERSION}/).to_stdout
    end

    it 'shows version with -v flag' do
      cli = described_class.new(['-v'])
      expect { cli.run }.to output(/fastembed #{Fastembed::VERSION}/).to_stdout.and raise_error(SystemExit)
    end
  end

  describe 'list-models command' do
    it 'lists all supported models' do
      cli = described_class.new(['list-models'])
      expect { cli.run }.to output(%r{BAAI/bge-small-en-v1.5}).to_stdout
    end

    it 'shows model dimensions' do
      cli = described_class.new(['list-models'])
      expect { cli.run }.to output(/Dimensions: 384/).to_stdout
    end

    it 'works with models alias' do
      cli = described_class.new(['models'])
      expect { cli.run }.to output(/Available models/).to_stdout
    end
  end

  describe 'embed command' do
    it 'embeds text from arguments' do
      cli = described_class.new(['embed', 'hello world'])
      output = capture_stdout { cli.run }
      result = JSON.parse(output)

      expect(result).to be_an(Array)
      expect(result.first['text']).to eq('hello world')
      expect(result.first['embedding']).to be_an(Array)
      expect(result.first['embedding'].length).to eq(384)
    end

    it 'embeds multiple texts' do
      cli = described_class.new(%w[embed hello world])
      output = capture_stdout { cli.run }
      result = JSON.parse(output)

      expect(result.length).to eq(2)
      expect(result[0]['text']).to eq('hello')
      expect(result[1]['text']).to eq('world')
    end

    it 'outputs ndjson format' do
      cli = described_class.new(['embed', '-f', 'ndjson', 'hello', 'world'])
      output = capture_stdout { cli.run }
      lines = output.strip.split("\n")

      expect(lines.length).to eq(2)
      expect(JSON.parse(lines[0])['text']).to eq('hello')
      expect(JSON.parse(lines[1])['text']).to eq('world')
    end

    it 'outputs csv format' do
      cli = described_class.new(['embed', '-f', 'csv', 'hello'])
      output = capture_stdout { cli.run }
      values = output.strip.split(',')

      expect(values.length).to eq(384)
      expect(values.first.to_f).to be_a(Float)
    end

    it 'shows help with --help' do
      cli = described_class.new(['embed', '--help'])
      expect { cli.run }.to output(/Usage: fastembed embed/).to_stdout.and raise_error(SystemExit)
    end

    it 'errors when no text provided' do
      cli = described_class.new(['embed'])
      # Simulate non-tty stdin
      allow($stdin).to receive(:tty?).and_return(true)
      expect { cli.run }.to output(/No text provided/).to_stderr.and raise_error(SystemExit)
    end
  end

  describe 'rerank command' do
    it 'reranks documents against a query' do
      cli = described_class.new(['rerank', '-q', 'What is Ruby?', 'Ruby is a programming language', 'The sky is blue'])
      output = capture_stdout { cli.run }
      result = JSON.parse(output)

      expect(result).to be_an(Array)
      expect(result.length).to eq(2)
      expect(result.first).to have_key('document')
      expect(result.first).to have_key('score')
      expect(result.first).to have_key('index')
      # Most relevant document should be first (about Ruby)
      expect(result.first['document']).to include('Ruby')
    end

    it 'returns top-k results' do
      cli = described_class.new(['rerank', '-q', 'programming', '-k', '1', 'Python', 'Java', 'cooking'])
      output = capture_stdout { cli.run }
      result = JSON.parse(output)

      expect(result.length).to eq(1)
    end

    it 'outputs ndjson format' do
      cli = described_class.new(['rerank', '-q', 'test', '-f', 'ndjson', 'doc1', 'doc2'])
      output = capture_stdout { cli.run }
      lines = output.strip.split("\n")

      expect(lines.length).to eq(2)
      expect(JSON.parse(lines.first)).to have_key('document')
    end

    it 'shows help with --help' do
      cli = described_class.new(['rerank', '--help'])
      expect { cli.run }.to output(/Usage: fastembed rerank/).to_stdout.and raise_error(SystemExit)
    end

    it 'errors when no query provided' do
      cli = described_class.new(['rerank', 'doc1', 'doc2'])
      expect { cli.run }.to output(/Query is required/).to_stderr.and raise_error(SystemExit)
    end

    it 'errors when no documents provided' do
      cli = described_class.new(['rerank', '-q', 'test'])
      allow($stdin).to receive(:tty?).and_return(true)
      expect { cli.run }.to output(/No documents provided/).to_stderr.and raise_error(SystemExit)
    end
  end

  describe 'unknown command' do
    it 'shows error for unknown command' do
      cli = described_class.new(['unknown'])
      expect { cli.run }.to output(/Unknown command/).to_stderr.and raise_error(SystemExit)
    end
  end

  # Helper to capture stdout
  def capture_stdout
    original = $stdout
    $stdout = StringIO.new
    yield
    $stdout.string
  ensure
    $stdout = original
  end
end
