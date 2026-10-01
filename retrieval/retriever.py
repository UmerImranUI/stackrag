import os
import sys

# Add project root to Python import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

from embeddings.embedder import Embedder
from vectorstore.qdrant_ingestion import (
    get_qdrant_client,
    get_collection_name,
)


load_dotenv()


TOP_K = 5


class Retriever:
    """
    Semantic retriever for StackRAG V1.

    Converts a user query into a 384-dimensional embedding
    and searches Qdrant using cosine similarity.
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

        self.embedder = Embedder()
        self.client = get_qdrant_client()
        self.collection_name = get_collection_name()

    def search(self, query: str, top_k: int | None = None):
        """
        Search Qdrant for semantically similar chunks.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        limit = top_k or self.top_k

        # Generate the query embedding using the exact
        # same model used when indexing the documents.
        query_vector = self.embedder.embed_text(query)

        # Validate query vector dimension.
        if len(query_vector) != self.embedder.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self.embedder.dimension}, "
                f"got {len(query_vector)}"
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return results.points


def print_results(query: str, results):
    """
    Print retrieval results in a readable format.
    """
    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    for index, result in enumerate(results, start=1):
        payload = result.payload

        print("\n" + "-" * 80)
        print(f"RESULT #{index}")
        print("-" * 80)

        print(f"Score:       {result.score:.4f}")
        print(f"Chunk ID:    {payload.get('chunk_id')}")
        print(f"Document ID: {payload.get('document_id')}")
        print(f"Title:       {payload.get('title')}")
        print(f"Tags:        {payload.get('tags')}")
        print(
            f"Question score: {payload.get('question_score')}"
        )
        print(
            f"Answer score:   {payload.get('answer_score')}"
        )
        print(
            f"Accepted:       {payload.get('is_accepted')}"
        )
        print(f"Source:      {payload.get('source_url')}")

        text = payload.get("text", "")

        print("\nText:")
        print(text[:1500])

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("StackRAG V1 - Retrieval Test")
    print("=" * 80)

    retriever = Retriever(top_k=TOP_K)

    print(f"Collection: {retriever.collection_name}")
    print(f"Embedding model: {retriever.embedder.model_name}")
    print(f"Embedding dimension: {retriever.embedder.dimension}")
    print(f"Top K: {TOP_K}")

    # Test queries.
    queries = [
        "How can I retry a failed Airflow task?",
        "How do I process large datasets with Apache Spark?",
        "How can I work with PySpark DataFrames?",
    ]

    for query in queries:
        results = retriever.search(query)
        print_results(query, results)


if __name__ == "__main__":
    main()