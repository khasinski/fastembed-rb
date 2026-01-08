RSpec.describe Fastembed do
  it "has a version number" do
    expect(Fastembed::VERSION).not_to be_nil
  end
end

RSpec.describe Fastembed::ModelInfo do
  describe "SUPPORTED_MODELS" do
    it "contains the default model" do
      expect(Fastembed::SUPPORTED_MODELS).to have_key(Fastembed::DEFAULT_MODEL)
    end

    it "has correct dimensions for bge-small-en-v1.5" do
      model = Fastembed::SUPPORTED_MODELS["BAAI/bge-small-en-v1.5"]
      expect(model.dim).to eq(384)
    end

    it "has correct dimensions for bge-base-en-v1.5" do
      model = Fastembed::SUPPORTED_MODELS["BAAI/bge-base-en-v1.5"]
      expect(model.dim).to eq(768)
    end

    it "has HuggingFace source for all models" do
      Fastembed::SUPPORTED_MODELS.each do |name, info|
        expect(info.hf_repo).not_to be_nil, "Model #{name} missing HF repo"
      end
    end
  end

  describe "#to_h" do
    it "converts model info to hash" do
      model = Fastembed::SUPPORTED_MODELS["BAAI/bge-small-en-v1.5"]
      hash = model.to_h

      expect(hash[:model_name]).to eq("BAAI/bge-small-en-v1.5")
      expect(hash[:dim]).to eq(384)
      expect(hash[:pooling]).to eq(:mean)
    end
  end
end

RSpec.describe Fastembed::Pooling do
  describe ".mean_pooling" do
    it "computes mean of token embeddings weighted by attention mask" do
      token_embeddings = [
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]
      ]
      attention_mask = [[1, 1, 0]]

      result = Fastembed::Pooling.mean_pooling(token_embeddings, attention_mask)

      expect(result.length).to eq(1)
      expect(result[0][0]).to eq(2.0) # (1 + 3) / 2
      expect(result[0][1]).to eq(3.0) # (2 + 4) / 2
    end
  end

  describe ".cls_pooling" do
    it "returns first token embedding" do
      token_embeddings = [
        [[1.0, 2.0], [3.0, 4.0]]
      ]
      attention_mask = [[1, 1]]

      result = Fastembed::Pooling.cls_pooling(token_embeddings, attention_mask)

      expect(result.length).to eq(1)
      expect(result[0]).to eq([1.0, 2.0])
    end
  end

  describe ".normalize" do
    it "L2 normalizes vectors" do
      vectors = [[3.0, 4.0]]

      result = Fastembed::Pooling.normalize(vectors)

      expect(result[0][0]).to be_within(0.001).of(0.6)
      expect(result[0][1]).to be_within(0.001).of(0.8)
    end

    it "handles zero vectors" do
      vectors = [[0.0, 0.0]]

      result = Fastembed::Pooling.normalize(vectors)

      expect(result[0]).to eq([0.0, 0.0])
    end
  end
end

RSpec.describe Fastembed::ModelManagement do
  describe ".cache_dir" do
    it "returns a path" do
      expect(Fastembed::ModelManagement.cache_dir).to be_a(String)
      expect(Fastembed::ModelManagement.cache_dir).to include("fastembed")
    end
  end

  describe ".resolve_model_info" do
    it "returns model info for valid model" do
      info = Fastembed::ModelManagement.resolve_model_info("BAAI/bge-small-en-v1.5")
      expect(info).to be_a(Fastembed::ModelInfo)
    end

    it "raises error for unknown model" do
      expect {
        Fastembed::ModelManagement.resolve_model_info("unknown/model")
      }.to raise_error(ArgumentError, /Unknown model/)
    end
  end

  describe ".model_directory" do
    it "creates safe directory name" do
      info = Fastembed::SUPPORTED_MODELS["BAAI/bge-small-en-v1.5"]
      dir = Fastembed::ModelManagement.model_directory(info)

      expect(dir).to include("BAAI--bge-small-en-v1.5")
      expect(dir).not_to include("/BAAI/")
    end
  end
end

RSpec.describe Fastembed::TextEmbedding do
  describe ".list_supported_models" do
    it "returns array of model hashes" do
      models = Fastembed::TextEmbedding.list_supported_models

      expect(models).to be_an(Array)
      expect(models).not_to be_empty
      expect(models.first).to have_key(:model_name)
      expect(models.first).to have_key(:dim)
    end
  end

  describe ".get_model_info" do
    it "returns model info for valid model" do
      info = Fastembed::TextEmbedding.get_model_info("BAAI/bge-small-en-v1.5")

      expect(info).to be_a(Hash)
      expect(info[:dim]).to eq(384)
    end

    it "returns nil for unknown model" do
      info = Fastembed::TextEmbedding.get_model_info("unknown/model")
      expect(info).to be_nil
    end
  end
end
