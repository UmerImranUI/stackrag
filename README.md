# StackRAG V1

A retrieval-augmented generation (RAG) system built around technical content from Stack Overflow.

StackRAG V1 extracts Stack Overflow questions and answers, stores the data in PostgreSQL, transforms answers into RAG documents, generates semantic embeddings using Sentence Transformers, stores vectors in Qdrant, retrieves relevant chunks, and uses a local Ollama LLM to generate grounded answers with source citations.

---

## Architecture

```text
                    Stack Overflow
                         │
                         ▼
              Stack Exchange API
                         │
                         ▼
                ┌─────────────────┐
                │   PostgreSQL    │
                │                 │
                │ Questions       │
                │ Answers         │
                │ RAG Documents   │
                └────────┬────────┘
                         │
                         ▼
              Cleaning & Chunking
                         │
                         ▼
              Sentence Transformers
              all-MiniLM-L6-v2
                         │
                         ▼
                ┌─────────────────┐
                │     Qdrant      │
                │                 │
                │ Vector Search   │
                │ Metadata        │
                └────────┬────────┘
                         │
                    User Query
                         │
                         ▼
                 Semantic Retrieval
                         │
                         ▼
                 Retrieved Context
                         │
                         ▼
                Grounded RAG Prompt
                         │
                         ▼
                   Ollama / LLM
                         │
                         ▼
                Answer + Sources
```

---

## V1 Pipeline

The current V1 pipeline consists of the following stages:

1. Extract Stack Overflow questions and answers through the Stack Exchange API.
2. Store questions and answers in PostgreSQL.
3. Build one RAG document per Stack Overflow answer.
4. Clean and combine question/answer content.
5. Split documents into chunks.
6. Generate embeddings using `all-MiniLM-L6-v2`.
7. Store embeddings and metadata in Qdrant.
8. Convert user queries into embeddings.
9. Retrieve the most semantically similar chunks from Qdrant.
10. Build a grounded prompt from the retrieved context.
11. Generate the final answer using a local Ollama model.
12. Return the answer together with the retrieved source metadata.

---

## Project Structure

```text
openrag_project/
│
├── app/
│
├── config/
│   └── settings.py
│
├── dags/
│
├── database/
│   ├── schema.sql
│   └── build_rag_docs.sql
│
├── embeddings/
│   ├── embedder.py
│   └── build_embeddings.py
│
├── evaluation/
│   ├── evaluate.py
│   └── questions.json
│
├── ingestion/
│   ├── script.py
│   └── run_sql.py
│
├── processing/
│   ├── cleaner.py
│   ├── chunker.py
│   └── __init__.py
│
├── rag/
│   ├── generator.py
│   ├── pipeline.py
│   └── prompt.py
│
├── retrieval/
│   └── retriever.py
│
├── vectorstore/
│   └── qdrant_ingestion.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component           | Technology                          |
| ------------------- | ----------------------------------- |
| Source              | Stack Overflow / Stack Exchange API |
| Language            | Python                              |
| Relational Database | PostgreSQL                          |
| Embeddings          | Sentence Transformers               |
| Embedding Model     | `all-MiniLM-L6-v2`                  |
| Vector Database     | Qdrant                              |
| LLM Runtime         | Ollama                              |
| LLM                 | `llama3.2:3b`                       |
| Similarity          | Cosine similarity                   |
| RAG                 | Custom Python pipeline              |

---

## PostgreSQL Data Model

The PostgreSQL layer contains:

### `questions`

Stores Stack Overflow questions including:

* Question ID
* Title
* Body
* Tags
* Score
* View count
* Answer count
* Accepted answer ID
* Creation/activity dates
* Source link
* Raw API JSON

### `answers`

Stores answers associated with questions including:

* Answer ID
* Question ID
* Answer body
* Score
* Accepted status
* Creation/activity dates
* Source link
* Raw API JSON

### `ingestion_runs`

Tracks ingestion executions and their status:

```text
RUNNING
SUCCESS
FAILED
```

It also records question/answer counts and error information.

### `rag_documents`

Creates the RAG-ready representation of Stack Overflow answers.

V1 creates **one RAG document per answer**.

Each document contains:

* Document ID
* Question ID
* Answer ID
* Title
* Question body
* Answer body
* Tags
* Question score
* Answer score
* Accepted-answer flag
* Source URL
* Creation timestamps

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stackrag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Stack Exchange API
STACK_API_KEY=your_stack_exchange_api_key
STACK_ACCESS_TOKEN=your_stack_exchange_access_token

# Qdrant
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=stackrag_v1
```

Do **not** commit `.env` or any API keys/secrets to Git.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd openrag_project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

The main Python dependencies are:

```text
requests
psycopg2-binary
python-dotenv
beautifulsoup4
sentence-transformers
qdrant-client
```

---

## PostgreSQL Setup

Create a PostgreSQL database named:

```text
stackrag
```

Then initialize the database schema:

```bash
psql -U postgres -d stackrag -f database/schema.sql
```

The schema creates the required questions, answers, ingestion-runs, and RAG-document tables.

---

## Data Ingestion

Configure the Stack Exchange API credentials in `.env`.

Then run:

```bash
python ingestion/script.py
```

The ingestion pipeline:

```text
Stack Exchange API
       ↓
Questions
       ↓
Answers
       ↓
PostgreSQL
```

The V1 ingestion process performs a full overwrite of the existing questions and answers before loading the newly extracted dataset.

---

## Build RAG Documents

After ingestion, build the RAG documents:

```bash
python ingestion/run_sql.py
```

This executes:

```text
database/build_rag_docs.sql
```

The SQL creates one RAG document for each Stack Overflow answer that contains a non-empty answer body.

---

## Generate Embeddings

Once RAG documents exist in PostgreSQL:

```bash
python embeddings/build_embeddings.py
```

This pipeline:

```text
PostgreSQL RAG Documents
          ↓
     Clean document
          ↓
        Chunk
          ↓
Sentence Transformer
          ↓
384-dimensional vector
          ↓
       Qdrant
```

The embedding model is:

```text
all-MiniLM-L6-v2
```

The resulting vectors have:

```text
Dimension: 384
Distance: Cosine
```

The Qdrant collection defaults to:

```text
stackrag_v1
```

---

## Retrieval

The retriever converts the user's question into an embedding and performs semantic similarity search against Qdrant.

The default retrieval size is:

```text
Top K = 5
```

Run the retrieval test with:

```bash
python retrieval/retriever.py
```

Example queries included in the retrieval test:

```text
How can I retry a failed Airflow task?

How do I process large datasets with Apache Spark?

How can I work with PySpark DataFrames?
```

---

## RAG Generation

V1 uses Ollama for local LLM generation.

The configured endpoint is:

```text
http://localhost:11434/api/generate
```

The default model is:

```text
llama3.2:3b
```

Make sure Ollama is running locally and the model is available before running the RAG pipeline.

Run:

```bash
python rag/pipeline.py
```

The complete flow is:

```text
User Query
    ↓
Query Embedding (all-MiniLM-L6-v2)
    ↓
Qdrant Semantic Search
    ↓
Top-K Retrieved Chunks
    ↓
Grounded Prompt
    ↓
Ollama
    ↓
Answer + Sources
```

---

## Grounding

The RAG prompt is designed around a strict grounding approach.

The retrieved sources are treated as the factual context for the generated response.

The model is instructed to:

* Use retrieved sources as the factual basis.
* Cite factual claims using `[Source N]`.
* Avoid unsupported claims.
* Avoid inventing information.
* State when the retrieved context does not contain enough information.

This allows the generated response to remain traceable to the retrieved Stack Overflow content.

---

## Evaluation

V1 includes a small evaluation dataset in:

```text
evaluation/questions.json
```

The evaluation currently measures:

### Retrieval

**Recall@K**

Checks whether an expected source appears within the top-K retrieved results.

### Generation

**Citation Presence**

Checks whether the generated answer contains a valid source citation.

**Key Concept Coverage**

Checks how many expected concepts appear lexically in the generated answer.

The evaluation script uses:

```text
TOP_K = 5
```

Run:

```bash
python evaluation/evaluate.py
```

---

## V1 Evaluation Snapshot

The included evaluation report was generated using 5 questions.

The recorded results were:

```text
Questions evaluated:       5

Retrieval:
Recall@5:                  100%

Generation:
Successful generations:   4 / 5
Citation presence:         50%
Key concept coverage:      62.5%
```

These numbers represent the included V1 evaluation run and should be treated as a baseline rather than a general performance guarantee.

---

## Example RAG Result

A successful query returns a structured result containing:

```python
{
    "query": "...",
    "answer": "...",
    "sources": [
        {
            "document_id": "...",
            "answer_id": 123,
            "chunk_id": "...",
            "chunk_index": 0,
            "title": "...",
            "url": "...",
            "score": 0.73,
            "text": "..."
        }
    ]
}
```

This makes the generated answer traceable to the retrieved Stack Overflow chunks.

---

## Current V1 Scope

V1 focuses on establishing the core RAG pipeline:

```text
Ingestion
   ↓
PostgreSQL
   ↓
RAG Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
   ↓
Retrieval
   ↓
Ollama
   ↓
Grounded Answer
```

The project structure also includes directories for planned orchestration and application components.

---

## Roadmap

Potential next stages include:

* Airflow orchestration
* Databricks Bronze/Silver/Gold processing
* Improved chunking strategies
* Better retrieval evaluation
* MRR and additional retrieval metrics
* Reranking
* Improved citation validation
* Automated tests
* Configuration management
* Logging and monitoring
* Streamlit interface
* End-to-end pipeline orchestration
* Demo deployment

---

## License

This project is licensed under the [MIT License](LICENSE).