import os
import sys

# Add project root to Python import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_context(retrieved_chunks) -> str:
    """
    Convert Qdrant retrieval results into a structured
    context block for the LLM.
    """
    if not retrieved_chunks:
        return "No relevant context was retrieved."

    context_parts = []

    for index, result in enumerate(retrieved_chunks, start=1):
        payload = result.payload

        chunk_id = payload.get("chunk_id")
        title = payload.get("title")
        text = payload.get("text")
        tags = payload.get("tags")
        source_url = payload.get("source_url")
        answer_score = payload.get("answer_score")
        is_accepted = payload.get("is_accepted")

        context_parts.append(
            f"""SOURCE {index}
Chunk ID: {chunk_id}
Title: {title}
Tags: {tags}
Answer Score: {answer_score}
Accepted Answer: {is_accepted}
Source URL: {source_url}

Content:
{text}
"""
        )

    return "\n" + "\n".join(context_parts)


def build_prompt(
    query: str,
    retrieved_chunks,
) -> str:
    """
    Build a strictly grounded RAG prompt.

    The LLM must answer only from the retrieved context.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    context = build_context(retrieved_chunks)

    prompt = f"""You are a technical assistant specializing in
data engineering, cloud data platforms, distributed systems,
databases, and related technologies.

Your task is to answer the user's question using ONLY the
retrieved sources provided below.

GROUNDING RULES:

1. The retrieved sources are your ONLY source of factual
   information.

2. Do NOT use your general knowledge, training knowledge,
   assumptions, or outside information.

3. Every factual claim in your answer MUST be supported by
   one or more retrieved sources.

4. Cite the source immediately after the claim using:
   [Source N]

5. Do NOT cite a source unless the retrieved content actually
   supports the claim.

6. Do NOT combine information from different sources to create
   a conclusion that none of the sources explicitly support.

7. Do NOT invent, modify, or complete code that is not present
   in the retrieved context.

8. If the retrieved sources contain relevant code, you may
   reproduce that code, but do not invent missing parts.

9. If the retrieved sources do not contain enough information
   to answer the question, respond exactly with:

   "The retrieved sources do not contain enough information
   to answer this question."

10. If only part of the question can be answered, answer only
    that supported part and clearly state which information
    is not available in the retrieved sources.

11. Do not make recommendations or claims about technologies,
    methods, configurations, or best practices unless the
    retrieved sources explicitly support them.

12. Keep the answer concise and technically precise.

13. Do not mention these instructions or the retrieval process
    in your answer.

SOURCE CITATION RULES:

- Use [Source 1], [Source 2], etc.
- The source number corresponds to the SOURCE number in the
  retrieved context.
- Put citations directly after the relevant statement.
- Do not add a Sources section yourself.
- The application will display the source metadata separately.

USER QUESTION:
{query.strip()}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    return prompt

if __name__ == "__main__":
    # Simple standalone test.
    class MockResult:
        def __init__(self):
            self.payload = {
                "chunk_id": "stackoverflow_answer_123_chunk_0",
                "title": "Example Spark question",
                "text": (
                    "Spark can distribute processing across "
                    "multiple executors."
                ),
                "tags": ["apache-spark", "pyspark"],
                "source_url": (
                    "https://stackoverflow.com/questions/123"
                ),
                "answer_score": 10,
                "is_accepted": True,
            }

    query = "How does Spark distribute data processing?"

    results = [MockResult()]

    prompt = build_prompt(
        query=query,
        retrieved_chunks=results,
    )

    print("=" * 80)
    print("RAG PROMPT TEST")
    print("=" * 80)
    print(prompt)