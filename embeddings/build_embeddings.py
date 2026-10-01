import os
import sys

# Add project root to Python import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import psycopg2
from dotenv import load_dotenv

from processing.cleaner import build_document_text
from processing.chunker import chunk_text
from embeddings.embedder import Embedder
from vectorstore.qdrant_ingestion import (
    VECTOR_DIMENSION,
    get_qdrant_client,
    get_collection_name,
    create_collection,
    upload_chunks,
)

load_dotenv()


BATCH_SIZE = 32


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def load_rag_documents():
    """
    Load all RAG documents from PostgreSQL.
    """
    query = """
        SELECT
            document_id,
            question_id,
            answer_id,
            title,
            question_body,
            answer_body,
            tags,
            question_score,
            answer_score,
            is_accepted,
            source_url
        FROM public.rag_documents
        ORDER BY document_id;
    """

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        conn.close()


def build_chunks(rows):
    """
    Clean and chunk every RAG document.

    Returns a list of dictionaries containing both
    the chunk text and its source metadata.
    """
    chunks = []

    for row in rows:
        (
            document_id,
            question_id,
            answer_id,
            title,
            question_body,
            answer_body,
            tags,
            question_score,
            answer_score,
            is_accepted,
            source_url,
        ) = row

        document_text = build_document_text(
            title=title,
            question_body=question_body,
            answer_body=answer_body,
        )

        document_chunks = chunk_text(document_text)

        for chunk_index, text in enumerate(document_chunks):
            chunks.append(
                {
                    "chunk_id": f"{document_id}_chunk_{chunk_index}",
                    "document_id": document_id,
                    "question_id": question_id,
                    "answer_id": answer_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "title": title,
                    "tags": tags,
                    "question_score": question_score,
                    "answer_score": answer_score,
                    "is_accepted": is_accepted,
                    "source_url": source_url,
                }
            )

    return chunks


def embed_chunks(chunks, embedder, batch_size=BATCH_SIZE):
    """
    Generate embeddings for chunks in batches.

    Embeddings are attached to the chunk dictionaries.
    """
    total_chunks = len(chunks)

    for start in range(0, total_chunks, batch_size):
        end = min(start + batch_size, total_chunks)

        batch = chunks[start:end]
        texts = [chunk["text"] for chunk in batch]

        vectors = embedder.embed_texts(texts)

        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector

        print(
            f"Embedded chunks: {end}/{total_chunks}"
        )

    return chunks


def main():
    print("=" * 80)
    print("StackRAG V1 - Build Embeddings")
    print("=" * 80)

    print("\nLoading RAG documents from PostgreSQL...")

    rows = load_rag_documents()

    if not rows:
        print("No RAG documents found.")
        return

    print(f"Documents loaded: {len(rows)}")

    print("\nCleaning and chunking documents...")

    chunks = build_chunks(rows)

    print(f"Chunks generated: {len(chunks)}")

    print("\nLoading embedding model...")

    embedder = Embedder()

    print(f"Model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.dimension}")
    print(f"Batch size: {BATCH_SIZE}")

    print("\nGenerating embeddings...")

    chunks = embed_chunks(
        chunks=chunks,
        embedder=embedder,
        batch_size=BATCH_SIZE,
    )

    # Basic validation.
    embedding_lengths = {
        len(chunk["embedding"])
        for chunk in chunks
    }

    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    print(f"Documents processed:     {len(rows)}")
    print(f"Chunks generated:        {len(chunks)}")
    print(f"Embeddings generated:    {len(chunks)}")
    print(f"Embedding dimensions:    {embedding_lengths}")
    print(f"Model:                   {embedder.model_name}")

    if len(embedding_lengths) == 1:
        dimension = next(iter(embedding_lengths))

        if dimension == 384:
            print("Dimension validation:    PASS")
        else:
            print(
                f"Dimension validation:    FAIL "
                f"(expected 384, got {dimension})"
            )
            return
    else:
        print(
            "Dimension validation:    FAIL "
            "(inconsistent embedding dimensions)"
        )
        return

    print("\nSample chunk:")
    print("-" * 80)
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Document ID: {chunks[0]['document_id']}")
    print(f"Title: {chunks[0]['title']}")
    print(f"Text length: {len(chunks[0]['text'])}")
    print(f"Vector length: {len(chunks[0]['embedding'])}")
    print(f"First 5 values: {chunks[0]['embedding'][:5]}")

    print("\nConnecting to Qdrant Cloud...")

    client = get_qdrant_client()

    collection_name = get_collection_name()

    print(f"Collection: {collection_name}")

    create_collection(
        client=client,
        collection_name=collection_name,
    )

    print("\nUploading embeddings to Qdrant...")

    upload_chunks(
        client=client,
        collection_name=collection_name,
        chunks=chunks,
        batch_size=BATCH_SIZE,
    )

    print("\nQdrant upload complete.")

    collection_info = client.get_collection(
        collection_name
    )

    print(
        f"Points in collection: "
        f"{collection_info.points_count}"
    )

    print("\n" + "=" * 80)
    print("QDRANT SUMMARY")
    print("=" * 80)
    print(f"Collection:              {collection_name}")
    print(f"Points uploaded:         {len(chunks)}")
    print(f"Vector dimensions:       {VECTOR_DIMENSION}")
    print("Distance:                COSINE")
    print("=" * 80)

if __name__ == "__main__":
    main()
