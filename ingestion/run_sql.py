import os
import sys

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


load_dotenv()


SCHEMA_FILE = os.path.join(
    PROJECT_ROOT,
    "database",
    "build_rag_docs.sql",
)


def load_schema():
    """
    Read the database schema SQL file.
    """

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()


def run_schema():
    """
    Execute database/build_rag_docs.sql against PostgreSQL.
    """

    host = os.getenv(
        "POSTGRES_HOST",
        "localhost",
    )

    port = int(
        os.getenv(
            "POSTGRES_PORT",
            "5432",
        )
    )

    database = os.getenv(
        "POSTGRES_DB",
        "stackrag",
    )

    user = os.getenv(
        "POSTGRES_USER",
        "postgres",
    )

    password = os.getenv(
        "POSTGRES_PASSWORD",
        "",
    )

    print("=" * 80)
    print("STACKRAG DATABASE SCHEMA")
    print("=" * 80)

    print(f"\nDatabase: {database}")
    print(f"Host: {host}:{port}")
    print(f"Schema file: {SCHEMA_FILE}")

    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(
            f"Schema file not found: {SCHEMA_FILE}"
        )

    sql = load_schema()

    if not sql.strip():
        raise ValueError(
            "build_rag_docs.sqls is empty."
        )

    connection = None

    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )

        connection.autocommit = False

        with connection.cursor() as cursor:
            cursor.execute(sql)

        connection.commit()

        print("\nSchema executed successfully.")

    except Exception:
        if connection:
            connection.rollback()

        print("\nSchema execution failed.")
        raise

    finally:
        if connection:
            connection.close()

    print("\nDatabase connection closed.")


if __name__ == "__main__":
    run_schema()