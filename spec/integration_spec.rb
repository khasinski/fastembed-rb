# frozen_string_literal: true

# Integration tests that download models and test actual embedding generation
# These tests are tagged with :integration and can be skipped in CI if needed
# Run with: bundle exec rspec --tag integration

RSpec.describe "Integration", :integration do
  describe Fastembed::TextEmbedding do
    let(:embedding) { Fastembed::TextEmbedding.new }

    describe "#embed" do
      it "generates embeddings for single text" do
        vectors = embedding.embed("Hello world").to_a

        expect(vectors.length).to eq(1)
        expect(vectors.first.length).to eq(384)
        expect(vectors.first).to all(be_a(Float))
      end

      it "generates embeddings for multiple texts" do
        documents = ["Hello world", "This is a test"]
        vectors = embedding.embed(documents).to_a

        expect(vectors.length).to eq(2)
        vectors.each do |vector|
          expect(vector.length).to eq(384)
        end
      end

      it "returns normalized vectors" do
        vectors = embedding.embed("Test document").to_a
        vector = vectors.first

        # L2 norm should be approximately 1.0
        norm = Math.sqrt(vector.sum { |v| v * v })
        expect(norm).to be_within(0.01).of(1.0)
      end

      it "returns an Enumerator" do
        result = embedding.embed(["Hello", "World"])
        expect(result).to be_an(Enumerator)
      end

      it "generates consistent embeddings" do
        text = "Consistent test"
        vectors1 = embedding.embed(text).to_a
        vectors2 = embedding.embed(text).to_a

        expect(vectors1).to eq(vectors2)
      end

      it "generates different embeddings for different texts" do
        vectors = embedding.embed(["Cats are animals", "Programming is fun"]).to_a

        expect(vectors[0]).not_to eq(vectors[1])
      end
    end

    describe "#query_embed" do
      it "adds query prefix" do
        # Query embeddings should be slightly different due to prefix
        query_vectors = embedding.query_embed("What is AI?").to_a
        plain_vectors = embedding.embed("What is AI?").to_a

        # They should be different because of the prefix
        expect(query_vectors.first).not_to eq(plain_vectors.first)
      end
    end

    describe "#passage_embed" do
      it "adds passage prefix" do
        passage_vectors = embedding.passage_embed("AI is artificial intelligence").to_a
        plain_vectors = embedding.embed("AI is artificial intelligence").to_a

        expect(passage_vectors.first).not_to eq(plain_vectors.first)
      end
    end

    describe "#dim" do
      it "returns the embedding dimension" do
        expect(embedding.dim).to eq(384)
      end
    end
  end

  describe "Semantic similarity" do
    let(:embedding) { Fastembed::TextEmbedding.new }

    def cosine_similarity(a, b)
      dot_product = a.zip(b).sum { |x, y| x * y }
      # Vectors are already normalized, so just return dot product
      dot_product
    end

    it "similar texts have higher similarity than dissimilar texts" do
      texts = [
        "I love programming in Ruby",
        "Ruby is my favorite programming language",
        "The weather is sunny today"
      ]
      vectors = embedding.embed(texts).to_a

      sim_related = cosine_similarity(vectors[0], vectors[1])
      sim_unrelated = cosine_similarity(vectors[0], vectors[2])

      expect(sim_related).to be > sim_unrelated
    end
  end
end
