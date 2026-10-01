import os
import sys
import uuid
# Add project root to Python import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


load_dotenv()


VECTOR_DIMENSION = 384
DEFAULT_COLLECTION = "stackrag_v1"


def get_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant Cloud client using credentials
    stored in environment variables.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL is not set")

    if not api_key:
        raise ValueError("QDRANT_API_KEY is not set")

    return QdrantClient(
        url=url,
        api_key=api_key,
    )


def get_collection_name() -> str:
    return os.getenv(
        "QDRANT_COLLECTION",
        DEFAULT_COLLECTION,
    )


def create_collection(
    client: QdrantClient,
    collection_name: str,
):
    """
    Create the StackRAG collection if it does not already exist.

    all-MiniLM-L6-v2 produces 384-dimensional embeddings,
    and cosine distance is used for semantic similarity.
    """
    if client.collection_exists(collection_name):
        print(
            f"Collection already exists: {collection_name}"
        )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: {collection_name}"
    )


def upload_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    batch_size: int = 32,
):
    """
    Upload embedded chunks and their metadata to Qdrant.
    """
    total = len(chunks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch = chunks[start:end]

        points = []

        for chunk in batch:
            points.append(
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            chunk["chunk_id"],
                        )
                    ),
                    vector=chunk["embedding"],
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "question_id": chunk["question_id"],
                        "answer_id": chunk["answer_id"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "title": chunk["title"],
                        "tags": chunk["tags"],
                        "question_score": chunk["question_score"],
                        "answer_score": chunk["answer_score"],
                        "is_accepted": chunk["is_accepted"],
                        "source_url": chunk["source_url"],
                    },
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        print(
            f"Uploaded points: {end}/{total}"
        )