import json
import os
import time
import uuid
from datetime import datetime, timezone

import psycopg2
import requests
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

STACK_API_URL = "https://api.stackexchange.com/2.3"

STACK_SITE = os.getenv("STACK_SITE", "stackoverflow")

TAGS = [
    tag.strip()
    for tag in os.getenv(
        "STACK_TAGS",
        "data-engineering,apache-spark,pyspark,apache-airflow,"
        "databricks,dbt,sql,postgresql,python,azure-data-factory",
    ).split(",")
    if tag.strip()
]

QUESTIONS_PER_TAG = int(
    os.getenv("QUESTIONS_PER_TAG", "2")
)

PAGE_SIZE = min(
    int(os.getenv("PAGE_SIZE", "100")),
    100,
)

REQUEST_DELAY_SECONDS = float(
    os.getenv("REQUEST_DELAY_SECONDS", "1.0")
)

STACK_API_KEY = os.getenv("STACK_API_KEY", "").strip()
STACK_ACCESS_TOKEN = os.getenv(
    "STACK_ACCESS_TOKEN",
    ""
).strip()

DB_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost"
)

DB_PORT = int(
    os.getenv("POSTGRES_PORT", "5432")
)

DB_NAME = os.getenv(
    "POSTGRES_DB",
    "stackrag"
)

DB_USER = os.getenv(
    "POSTGRES_USER",
    "postgres"
)

DB_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
    ""
)


# ============================================================
# HTTP Session
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": "StackRAG/1.0"
    }
)


# ============================================================
# Stack Exchange API
# ============================================================

def api_get(endpoint, params):
    """
    Make a GET request to the Stack Exchange API.

    Uses:
    - application key when configured
    - access token when configured
    - API-provided backoff
    - quota logging
    - clear error handling

    Stack Exchange API errors are returned as HTTP 400
    with error details in the JSON response.
    """

    url = f"{STACK_API_URL}/{endpoint}"

    params = params.copy()

    # Application key.
    if STACK_API_KEY:
        params["key"] = STACK_API_KEY

    # OAuth access token.
    #
    # The API documentation recommends authentication
    # information via the Authorization header.
    headers = {}


    while True:

        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=30,
        )

        # ----------------------------------------------------
        # Parse response
        # ----------------------------------------------------

        try:
            data = response.json()
        except ValueError:
            data = {}

        # ----------------------------------------------------
        # API error
        # ----------------------------------------------------

        if not response.ok:

            error_name = data.get(
                "error_name",
                "unknown_error"
            )

            error_message = data.get(
                "error_message",
                response.text
            )

            error_id = data.get(
                "error_id",
                "unknown"
            )

            print()
            print("=" * 60)
            print("STACK EXCHANGE API ERROR")
            print("=" * 60)
            print(f"HTTP status : {response.status_code}")
            print(f"Error ID    : {error_id}")
            print(f"Error name  : {error_name}")
            print(f"Message     : {error_message}")
            print("=" * 60)

            # ------------------------------------------------
            # Throttle violation
            # ------------------------------------------------

            if error_name == "throttle_violation":

                print()
                print(
                    "Stack Exchange has throttled this API client/IP."
                )
                print(
                    "Do not retry immediately."
                )
                print(
                    "Use an application key/access token and "
                    "wait for the current throttle to clear."
                )
                print()

            response.raise_for_status()

        # ----------------------------------------------------
        # API backoff
        # ----------------------------------------------------

        backoff = data.get("backoff")

        if backoff:

            print(
                f"API requested backoff: "
                f"waiting {backoff} seconds..."
            )

            time.sleep(backoff)

            continue

        # ----------------------------------------------------
        # Quota information
        # ----------------------------------------------------

        quota_remaining = data.get(
            "quota_remaining"
        )

        quota_max = data.get(
            "quota_max"
        )

        if quota_remaining is not None:

            print(
                f"  API quota: "
                f"{quota_remaining}/{quota_max}"
            )

        return data


# ============================================================
# Extract Questions
# ============================================================

def extract_questions():
    """
    Extract questions for the configured Stack Overflow tags.

    Each tag is limited to QUESTIONS_PER_TAG.

    Questions are deduplicated globally because the same
    question can contain multiple configured tags.

    Anonymous API access has a maximum page number of 25,
    so this function never requests page 26.
    """

    questions = {}

    for tag in TAGS:

        print()
        print(
            f"Extracting questions for tag: {tag}"
        )

        page = 1

        while page <= 25:

            # ------------------------------------------------
            # Count questions belonging to this tag
            # ------------------------------------------------

            tag_question_count = sum(
                1
                for question in questions.values()
                if tag in question.get("tags", [])
            )

            if tag_question_count >= QUESTIONS_PER_TAG:
                break

            # ------------------------------------------------
            # API parameters
            # ------------------------------------------------

            params = {
                "site": STACK_SITE,
                "tagged": tag,
                "page": page,
                "pagesize": PAGE_SIZE,
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
            }

            print(
                f"  Requesting page {page} "
                f"(pagesize={PAGE_SIZE})"
            )

            data = api_get(
                "questions",
                params,
            )

            items = data.get(
                "items",
                []
            )

            if not items:
                break

            # ------------------------------------------------
            # Add questions
            # ------------------------------------------------

            for question in items:

                question_id = question[
                    "question_id"
                ]

                questions[
                    question_id
                ] = question

            # ------------------------------------------------
            # Check target
            # ------------------------------------------------

            tag_question_count = sum(
                1
                for question in questions.values()
                if tag in question.get("tags", [])
            )

            if tag_question_count >= QUESTIONS_PER_TAG:
                break

            # ------------------------------------------------
            # Pagination
            # ------------------------------------------------

            if not data.get("has_more"):
                break

            page += 1

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        final_count = sum(
            1
            for question in questions.values()
            if tag in question.get("tags", [])
        )

        print(
            f"  Collected {final_count} "
            f"questions for tag: {tag}"
        )

        if page >= 25:
            print(
                f"  Reached API page limit for {tag}."
            )

    print()
    print(
        f"Unique questions extracted: "
        f"{len(questions)}"
    )

    return list(
        questions.values()
    )


# ============================================================
# Extract Answers
# ============================================================

def extract_answers(questions):
    """
    Extract answers for the extracted questions.

    Individual answer failures do not fail the entire
    ingestion run.
    """

    answers = {}

    total_questions = len(
        questions
    )

    failed_questions = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        question_id = question[
            "question_id"
        ]

        print()
        print(
            f"Extracting answers "
            f"{index}/{total_questions} "
            f"for question {question_id}"
        )

        page = 1

        while page <= 25:

            params = {
                "site": STACK_SITE,
                "page": page,
                "pagesize": PAGE_SIZE,
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
            }

            try:

                data = api_get(
                    f"questions/{question_id}/answers",
                    params,
                )

            except requests.exceptions.HTTPError as exc:

                print(
                    f"  WARNING: Failed to extract "
                    f"answers for question "
                    f"{question_id}: {exc}"
                )

                failed_questions.append(
                    {
                        "question_id": question_id,
                        "error": str(exc),
                    }
                )

                break

            items = data.get(
                "items",
                []
            )

            for answer in items:

                answers[
                    answer["answer_id"]
                ] = answer

            if not data.get(
                "has_more"
            ):
                break

            page += 1

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    print()
    print(
        f"Unique answers extracted: "
        f"{len(answers)}"
    )

    if failed_questions:

        print(
            f"Questions with failed "
            f"answer extraction: "
            f"{len(failed_questions)}"
        )

        for failure in failed_questions[:20]:

            print(
                f"  Question "
                f"{failure['question_id']}: "
                f"{failure['error']}"
            )

    return list(
        answers.values()
    )


# ============================================================
# PostgreSQL
# ============================================================

def get_db_connection():

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def to_datetime(timestamp):
    """
    Convert Stack Exchange Unix timestamp
    to timezone-aware UTC datetime.
    """

    if timestamp is None:
        return None

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    )


# ============================================================
# Insert Questions
# ============================================================

def insert_questions(
    cursor,
    questions,
):

    sql = """
        INSERT INTO public.questions (
            question_id,
            title,
            body,
            tags,
            score,
            view_count,
            answer_count,
            accepted_answer_id,
            owner_display_name,
            creation_date,
            last_activity_date,
            link,
            is_answered,
            raw_json
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (question_id)
        DO UPDATE SET
            title = EXCLUDED.title,
            body = EXCLUDED.body,
            tags = EXCLUDED.tags,
            score = EXCLUDED.score,
            view_count = EXCLUDED.view_count,
            answer_count = EXCLUDED.answer_count,
            accepted_answer_id =
                EXCLUDED.accepted_answer_id,
            owner_display_name =
                EXCLUDED.owner_display_name,
            creation_date =
                EXCLUDED.creation_date,
            last_activity_date =
                EXCLUDED.last_activity_date,
            link = EXCLUDED.link,
            is_answered =
                EXCLUDED.is_answered,
            raw_json =
                EXCLUDED.raw_json,
            inserted_at = NOW();
    """

    for question in questions:

        owner = question.get(
            "owner",
            {}
        )

        cursor.execute(
            sql,
            (
                question["question_id"],
                question.get("title"),
                question.get("body"),
                question.get("tags", []),
                question.get("score"),
                question.get("view_count"),
                question.get("answer_count"),
                question.get(
                    "accepted_answer_id"
                ),
                owner.get(
                    "display_name"
                ),
                to_datetime(
                    question.get(
                        "creation_date"
                    )
                ),
                to_datetime(
                    question.get(
                        "last_activity_date"
                    )
                ),
                question.get("link"),
                question.get(
                    "is_answered"
                ),
                json.dumps(question),
            ),
        )


# ============================================================
# Insert Answers
# ============================================================

def insert_answers(
    cursor,
    answers,
):

    sql = """
        INSERT INTO public.answers (
            answer_id,
            question_id,
            body,
            score,
            is_accepted,
            owner_display_name,
            creation_date,
            last_activity_date,
            link,
            raw_json
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (answer_id)
        DO UPDATE SET
            question_id =
                EXCLUDED.question_id,
            body =
                EXCLUDED.body,
            score =
                EXCLUDED.score,
            is_accepted =
                EXCLUDED.is_accepted,
            owner_display_name =
                EXCLUDED.owner_display_name,
            creation_date =
                EXCLUDED.creation_date,
            last_activity_date =
                EXCLUDED.last_activity_date,
            link =
                EXCLUDED.link,
            raw_json =
                EXCLUDED.raw_json,
            inserted_at = NOW();
    """

    for answer in answers:

        owner = answer.get(
            "owner",
            {}
        )

        cursor.execute(
            sql,
            (
                answer["answer_id"],
                answer["question_id"],
                answer.get("body"),
                answer.get("score"),
                answer.get(
                    "is_accepted",
                    False
                ),
                owner.get(
                    "display_name"
                ),
                to_datetime(
                    answer.get(
                        "creation_date"
                    )
                ),
                to_datetime(
                    answer.get(
                        "last_activity_date"
                    )
                ),
                answer.get("link"),
                json.dumps(answer),
            ),
        )


# ============================================================
# Ingestion Run
# ============================================================

def create_ingestion_run(
    cursor,
):

    run_id = str(
        uuid.uuid4()
    )

    cursor.execute(
        """
        INSERT INTO public.ingestion_runs (
            run_id,
            started_at,
            status
        )
        VALUES (%s, %s, %s);
        """,
        (
            run_id,
            datetime.now(
                timezone.utc
            ),
            "RUNNING",
        ),
    )

    return run_id


def complete_ingestion_run(
    cursor,
    run_id,
    status,
    questions_count,
    answers_count,
    error_message=None,
):

    cursor.execute(
        """
        UPDATE public.ingestion_runs
        SET
            completed_at = %s,
            status = %s,
            questions_extracted = %s,
            answers_extracted = %s,
            error_message = %s
        WHERE run_id = %s;
        """,
        (
            datetime.now(
                timezone.utc
            ),
            status,
            questions_count,
            answers_count,
            error_message,
            run_id,
        ),
    )


# ============================================================
# Full Overwrite
# ============================================================

def overwrite_database(cursor):

    print()
    print(
        "Overwriting existing PostgreSQL data..."
    )

    cursor.execute(
        """
        TRUNCATE TABLE
            public.answers,
            public.questions
        CASCADE;
        """
    )

    print(
        "Existing questions and answers removed."
    )


# ============================================================
# Main Pipeline
# ============================================================

def main():

    print("=" * 60)
    print(
        "StackRAG V1 - Stack Exchange Ingestion"
    )
    print("=" * 60)

    print(
        f"Site: {STACK_SITE}"
    )

    print(
        f"Tags: {', '.join(TAGS)}"
    )

    print(
        f"Questions per tag: "
        f"{QUESTIONS_PER_TAG}"
    )

    print(
        f"Page size: {PAGE_SIZE}"
    )

    print(
        f"Request delay: "
        f"{REQUEST_DELAY_SECONDS}s"
    )

    print(
        f"API key configured: "
        f"{bool(STACK_API_KEY)}"
    )

    print(
        f"Access token configured: "
        f"{bool(STACK_ACCESS_TOKEN)}"
    )

    # --------------------------------------------------------
    # Database connection
    # --------------------------------------------------------

    connection = get_db_connection()

    cursor = connection.cursor()

    run_id = create_ingestion_run(
        cursor
    )

    connection.commit()

    questions = []

    answers = []

    try:

        # ----------------------------------------------------
        # Extract questions
        # ----------------------------------------------------

        questions = extract_questions()

        # ----------------------------------------------------
        # Extract answers
        # ----------------------------------------------------

        answers = extract_answers(
            questions
        )

        # ----------------------------------------------------
        # Full overwrite
        # ----------------------------------------------------

        overwrite_database(
            cursor
        )

        # ----------------------------------------------------
        # Load questions
        # ----------------------------------------------------

        print()
        print(
            "Loading questions into PostgreSQL..."
        )

        insert_questions(
            cursor,
            questions,
        )

        print(
            f"Inserted {len(questions)} "
            f"questions."
        )

        # ----------------------------------------------------
        # Load answers
        # ----------------------------------------------------

        print()
        print(
            "Loading answers into PostgreSQL..."
        )

        insert_answers(
            cursor,
            answers,
        )

        print(
            f"Inserted {len(answers)} "
            f"answers."
        )

        # ----------------------------------------------------
        # Complete run
        # ----------------------------------------------------

        complete_ingestion_run(
            cursor,
            run_id,
            "SUCCESS",
            len(questions),
            len(answers),
        )

        connection.commit()

        print()
        print("=" * 60)
        print(
            "INGESTION COMPLETED SUCCESSFULLY"
        )
        print("=" * 60)

    except Exception as exc:

        connection.rollback()

        print()
        print("=" * 60)
        print(
            "INGESTION FAILED"
        )
        print("=" * 60)

        print(
            str(exc)
        )

        cursor = connection.cursor()

        complete_ingestion_run(
            cursor,
            run_id,
            "FAILED",
            len(questions),
            len(answers),
            str(exc),
        )

        connection.commit()

        raise

    finally:

        cursor.close()

        connection.close()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()