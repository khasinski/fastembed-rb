# frozen_string_literal: true

require_relative "lib/fastembed/version"

Gem::Specification.new do |spec|
  spec.name          = "fastembed"
  spec.version       = Fastembed::VERSION
  spec.authors       = ["Chris Hasinski"]
  spec.email         = ["krzysztof.hasinski@gmail.com"]

  spec.summary       = "Fast, lightweight text embeddings for Ruby"
  spec.description   = "A Ruby port of FastEmbed - a lightweight, fast library for generating text embeddings using ONNX Runtime"
  spec.homepage      = "https://github.com/hasik/fastembed-rb"
  spec.license       = "MIT"
  spec.required_ruby_version = ">= 3.0.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) ||
        f.start_with?(*%w[bin/ test/ spec/ features/ .git .github])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_dependency "onnxruntime", "~> 0.9"
  spec.add_dependency "tokenizers", "~> 0.5"

  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rspec", "~> 3.0"
end
