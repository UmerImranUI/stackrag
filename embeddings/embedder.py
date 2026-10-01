
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    Local text embedding model for StackRAG V1.

    Uses Sentence Transformers and runs entirely locally.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate a single embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return vectors.tolist()

    @property
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """
        return self.model.get_embedding_dimension()


if __name__ == "__main__":
    embedder = Embedder()

    text = """
    Apache Spark is a distributed data processing engine
    used for large-scale data transformations.
    """

    vector = embedder.embed_text(text)

    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
