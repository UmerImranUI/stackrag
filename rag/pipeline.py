import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.retriever import Retriever
from rag.prompt import build_prompt
from rag.generator import Generator


class RAGPipeline:
    """
    End-to-end StackRAG pipeline.

    Query
      ↓
    Retrieval
      ↓
    Prompt construction
      ↓
    Local LLM generation
      ↓
    Structured result
    """

    def __init__(
        self,
        top_k: int = 5,
    ):
        self.retriever = Retriever(top_k=top_k)
        self.generator = Generator()

    def ask(self, query: str) -> dict:
        """
        Run a complete RAG query and return a structured result.

        Returns:
            {
                "query": str,
                "answer": str,
                "sources": [
                    {
                        "document_id": str,
                        "answer_id": int,
                        "chunk_id": str,
                        "chunk_index": int,
                        "title": str,
                        "url": str,
                        "score": float,
                        "text": str
                    }
                ]
            }
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # 1. Retrieve relevant chunks.
        retrieved_chunks = self.retriever.search(query)
        
        

        # 2. Build grounded prompt.
        prompt = build_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        # 3. Generate answer using local LLM.
        answer = self.generator.generate(prompt)

        # 4. Extract source metadata and retrieved chunk text.
        sources = []

        for result in retrieved_chunks:
            payload = result.payload or {}

            sources.append(
                {
                    "document_id": payload.get("document_id"),
                    "answer_id": payload.get("answer_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "title": payload.get("title"),
                    "url": payload.get("source_url"),
                    "score": round(float(result.score), 4),
                    "text": payload.get("text"),
                }
            )

        return {
            "query": query.strip(),
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=3)

    query = "How do I extract tables from invoices with fixed column widths using pdfplumber?"

    print("=" * 80)
    print("STACKRAG V1")
    print("=" * 80)

    print(f"\nQUESTION:\n{query}")

    print("\nGenerating answer...\n")

    result = pipeline.ask(query)

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(f"\n[Source {index}]")
        print(f"Document ID: {source['document_id']}")
        print(f"Answer ID: {source['answer_id']}")
        print(f"Chunk ID: {source['chunk_id']}")
        print(f"Chunk Index: {source['chunk_index']}")
        # print(f"Title: {source['title']}")
        print(f"URL: {source['url']}")
        print(f"Similarity Score: {source['score']}")
        # print("\nRetrieved Chunk:")
        # print(source["text"])