import json
import os
import sys
import re
from datetime import datetime, timezone


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from rag.pipeline import RAGPipeline
from rag.prompt import build_prompt


QUESTIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "questions.json",
)

REPORT_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "evaluation_report.json",
)

TOP_K = 5


def load_questions() -> list[dict]:
    """
    Load evaluation questions from questions.json.
    """

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    """
    Normalize text for simple concept matching.
    """

    if not text:
        return ""

    text = text.lower()

    # Keep characters useful for technical terms such as:
    # PySpark, spark.read.json, extract_table, etc.
    text = re.sub(
        r"[^a-z0-9+#._-]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def check_recall_at_k(
    retrieved_sources: list[dict],
    expected_source_ids: list[str],
) -> tuple[bool, list[str], int | None]:
    """
    Check whether an expected source appears in the
    retrieved top-K results.

    Returns:
        passed,
        matched_source_ids,
        first_matching_rank
    """

    retrieved_ids = []

    for source in retrieved_sources:
        document_id = source.get("document_id")

        if document_id and document_id not in retrieved_ids:
            retrieved_ids.append(document_id)

    matched_source_ids = [
        source_id
        for source_id in expected_source_ids
        if source_id in retrieved_ids
    ]

    rank = None

    for index, document_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if document_id in expected_source_ids:
            rank = index
            break

    return (
        bool(matched_source_ids),
        matched_source_ids,
        rank,
    )


def check_citation_presence(
    answer: str,
    source_count: int,
) -> tuple[bool, list[int]]:
    """
    Check whether the generated answer contains at least
    one valid [Source N] citation.

    Returns:
        passed,
        valid citation numbers
    """

    if not answer:
        return False, []

    citations = re.findall(
        r"\[Source\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    )

    valid_citations = []

    for number in citations:
        source_number = int(number)

        if 1 <= source_number <= source_count:
            if source_number not in valid_citations:
                valid_citations.append(source_number)

    return (
        bool(valid_citations),
        valid_citations,
    )


def check_key_concepts(
    answer: str,
    key_concepts: list[str],
) -> tuple[int, int, list[str], list[str]]:
    """
    Check how many expected concepts appear in the answer.

    This is a simple lexical coverage check.
    It is NOT a true faithfulness/groundedness metric.
    """

    normalized_answer = normalize_text(answer)

    matched = []
    missing = []

    for concept in key_concepts:
        normalized_concept = normalize_text(concept)

        if normalized_concept in normalized_answer:
            matched.append(concept)
        else:
            missing.append(concept)

    return (
        len(matched),
        len(key_concepts),
        matched,
        missing,
    )


def retrieve(
    pipeline: RAGPipeline,
    question: str,
) -> list[dict]:
    """
    Run retrieval only.

    This deliberately does NOT call the LLM.
    """

    retrieved_chunks = pipeline.retriever.search(
        question,
        top_k=TOP_K,
    )

    sources = []

    for result in retrieved_chunks:
        payload = result.payload or {}

        sources.append(
            {
                "document_id": payload.get(
                    "document_id"
                ),
                "answer_id": payload.get(
                    "answer_id"
                ),
                "chunk_id": payload.get(
                    "chunk_id"
                ),
                "chunk_index": payload.get(
                    "chunk_index"
                ),
                "title": payload.get(
                    "title"
                ),
                "url": payload.get(
                    "source_url"
                ),
                "score": round(
                    float(result.score),
                    4,
                ),
                "text": payload.get(
                    "text"
                ),
            }
        )

    return sources


def generate_answer(
    pipeline: RAGPipeline,
    question: str,
    sources: list[dict],
) -> str:
    """
    Generate an answer using the already retrieved sources.

    Retrieval is NOT repeated.
    """

    # Reconstruct the Qdrant-style objects required by
    # build_prompt() from the structured source data.

    class RetrievedResult:
        def __init__(self, source):
            self.score = source["score"]
            self.payload = {
                "document_id": source["document_id"],
                "answer_id": source["answer_id"],
                "chunk_id": source["chunk_id"],
                "chunk_index": source["chunk_index"],
                "title": source["title"],
                "tags": [],
                "answer_score": None,
                "is_accepted": False,
                "source_url": source["url"],
                "text": source["text"],
            }

    retrieved_results = [
        RetrievedResult(source)
        for source in sources
    ]

    prompt = build_prompt(
        query=question,
        retrieved_chunks=retrieved_results,
    )

    return pipeline.generator.generate(prompt)


def evaluate_question(
    pipeline: RAGPipeline,
    question_data: dict,
) -> dict:

    question_id = question_data["id"]
    question = question_data["question"]

    expected_source_ids = question_data[
        "expected_source_ids"
    ]

    key_concepts = question_data[
        "key_concepts"
    ]

    print("\n" + "=" * 80)
    print(f"{question_id}: {question}")
    print("=" * 80)

    result = {
        "id": question_id,
        "question": question,
        "expected_source_ids": expected_source_ids,
        "key_concepts": key_concepts,
        "retrieval": {},
        "generation": {},
    }

    # ------------------------------------------------------------------
    # 1. RETRIEVAL EVALUATION
    # ------------------------------------------------------------------

    try:
        sources = retrieve(
            pipeline,
            question,
        )

        (
            retrieval_pass,
            matched_source_ids,
            rank,
        ) = check_recall_at_k(
            retrieved_sources=sources,
            expected_source_ids=expected_source_ids,
        )

        result["retrieval"] = {
            "success": True,
            "recall_at_k": retrieval_pass,
            "matched_source_ids": matched_source_ids,
            "first_matching_rank": rank,
            "retrieved_sources": sources,
        }

        print(
            f"\nRecall@{TOP_K}: "
            f"{'PASS' if retrieval_pass else 'FAIL'}"
        )

        print(
            f"Matched Sources: "
            f"{matched_source_ids}"
        )

        if rank is not None:
            print(
                f"Expected source rank: {rank}"
            )

    except Exception as error:

        result["retrieval"] = {
            "success": False,
            "recall_at_k": False,
            "error": str(error),
        }

        print(
            f"\nRetrieval ERROR: {error}"
        )

        # Cannot perform generation without retrieval.
        result["generation"] = {
            "success": False,
            "citation_present": False,
            "concepts_matched": 0,
            "concepts_total": len(key_concepts),
            "concept_coverage": 0.0,
            "error": (
                "Generation skipped because "
                "retrieval failed."
            ),
        }

        return result

    # ------------------------------------------------------------------
    # 2. GENERATION EVALUATION
    # ------------------------------------------------------------------

    try:
        answer = generate_answer(
            pipeline=pipeline,
            question=question,
            sources=sources,
        )

        citation_pass, citations = (
            check_citation_presence(
                answer=answer,
                source_count=len(sources),
            )
        )

        (
            concepts_matched,
            concepts_total,
            matched_concepts,
            missing_concepts,
        ) = check_key_concepts(
            answer=answer,
            key_concepts=key_concepts,
        )

        concept_coverage = (
            concepts_matched / concepts_total
            if concepts_total
            else 0.0
        )

        result["generation"] = {
            "success": True,
            "citation_present": citation_pass,
            "citations": citations,
            "concepts_matched": concepts_matched,
            "concepts_total": concepts_total,
            "concept_coverage": concept_coverage,
            "matched_concepts": matched_concepts,
            "missing_concepts": missing_concepts,
            "answer": answer,
        }

        print(
            f"Citation Presence: "
            f"{'PASS' if citation_pass else 'FAIL'}"
        )

        print(
            f"Key Concept Coverage: "
            f"{concepts_matched}/{concepts_total} "
            f"({concept_coverage:.0%})"
        )

        if missing_concepts:
            print(
                f"Missing Concepts: "
                f"{missing_concepts}"
            )

    except Exception as error:

        result["generation"] = {
            "success": False,
            "citation_present": False,
            "concepts_matched": 0,
            "concepts_total": len(key_concepts),
            "concept_coverage": 0.0,
            "error": str(error),
        }

        print(
            f"Generation ERROR: {error}"
        )

    return result


def build_summary(
    results: list[dict],
) -> dict:

    total = len(results)

    retrieval_successes = sum(
        result["retrieval"].get(
            "success",
            False,
        )
        for result in results
    )

    retrieval_passes = sum(
        result["retrieval"].get(
            "recall_at_k",
            False,
        )
        for result in results
    )

    generation_successes = sum(
        result["generation"].get(
            "success",
            False,
        )
        for result in results
    )

    citation_passes = sum(
        result["generation"].get(
            "citation_present",
            False,
        )
        for result in results
    )

    total_concepts = sum(
        result["generation"].get(
            "concepts_total",
            0,
        )
        for result in results
    )

    matched_concepts = sum(
        result["generation"].get(
            "concepts_matched",
            0,
        )
        for result in results
    )

    return {
        "questions_evaluated": total,
        "retrieval": {
            "successful_questions": retrieval_successes,
            "recall_at_5_passes": retrieval_passes,
            "recall_at_5": (
                retrieval_passes / total
                if total
                else 0.0
            ),
        },
        "generation": {
            "successful_questions": generation_successes,
            "citation_passes": citation_passes,
            "citation_presence_rate": (
                citation_passes / generation_successes
                if generation_successes
                else 0.0
            ),
            "matched_concepts": matched_concepts,
            "total_concepts": total_concepts,
            "key_concept_coverage": (
                matched_concepts / total_concepts
                if total_concepts
                else 0.0
            ),
        },
    }


def print_summary(
    results: list[dict],
    summary: dict,
):
    """
    Print aggregate evaluation metrics.
    """

    print("\n")
    print("=" * 80)
    print("STACKRAG V1 EVALUATION SUMMARY")
    print("=" * 80)

    total = summary["questions_evaluated"]

    retrieval = summary["retrieval"]
    generation = summary["generation"]

    print(
        f"\nQuestions Evaluated: "
        f"{total}"
    )

    print(
        f"\nRetrieval"
    )

    print(
        f"  Successful Retrievals: "
        f"{retrieval['successful_questions']}/{total}"
    )

    print(
        f"  Recall@{TOP_K}: "
        f"{retrieval['recall_at_5_passes']}/{total} "
        f"({retrieval['recall_at_5']:.1%})"
    )

    print(
        f"\nGeneration"
    )

    print(
        f"  Successful Generations: "
        f"{generation['successful_questions']}/{total}"
    )

    print(
        f"  Citation Presence: "
        f"{generation['citation_passes']}/"
        f"{generation['successful_questions']} "
        f"({generation['citation_presence_rate']:.1%})"
    )

    print(
        f"  Key Concept Coverage: "
        f"{generation['matched_concepts']}/"
        f"{generation['total_concepts']} "
        f"({generation['key_concept_coverage']:.1%})"
    )

    print("\nPer-question results:")
    print("-" * 80)

    for result in results:

        retrieval = result["retrieval"]
        generation = result["generation"]

        retrieval_status = (
            "PASS"
            if retrieval.get(
                "recall_at_k",
                False,
            )
            else "FAIL"
        )

        if generation.get("success"):
            citation_status = (
                "PASS"
                if generation.get(
                    "citation_present",
                    False,
                )
                else "FAIL"
            )

            concepts = (
                f"{generation.get('concepts_matched', 0)}/"
                f"{generation.get('concepts_total', 0)}"
            )

        else:
            citation_status = "ERROR"
            concepts = "N/A"

        print(
            f"{result['id']} | "
            f"Retrieval: {retrieval_status} | "
            f"Citation: {citation_status} | "
            f"Concepts: {concepts}"
        )


def save_report(
    results: list[dict],
    summary: dict,
):
    """
    Save complete evaluation results as JSON.
    """

    report = {
        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "configuration": {
            "top_k": TOP_K,
            "questions_file": QUESTIONS_FILE,
        },

        "summary": summary,

        "results": results,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nJSON report saved to:"
        f"\n{REPORT_FILE}"
    )


def main():

    questions = load_questions()

    print("=" * 80)
    print("STACKRAG V1 EVALUATION")
    print("=" * 80)

    print(
        f"\nLoaded {len(questions)} "
        f"evaluation questions."
    )

    print(
        f"Retrieval configuration: "
        f"Recall@{TOP_K}"
    )

    pipeline = RAGPipeline(
        top_k=TOP_K
    )

    results = []

    for question_data in questions:

        result = evaluate_question(
            pipeline=pipeline,
            question_data=question_data,
        )

        results.append(result)

    summary = build_summary(
        results
    )

    print_summary(
        results,
        summary,
    )

    save_report(
        results,
        summary,
    )


if __name__ == "__main__":
    main()