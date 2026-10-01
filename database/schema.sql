-- ============================================================
-- StackRAG V1 - PostgreSQL Schema
-- Schema: public
-- Mode: Full overwrite
-- Source: Stack Exchange API
-- ============================================================

-- ------------------------------------------------------------
-- Questions
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.questions (
    question_id          BIGINT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    tags                 TEXT[],
    score                INTEGER,
    view_count           INTEGER,
    answer_count         INTEGER,
    accepted_answer_id   BIGINT,
    owner_display_name   TEXT,
    creation_date        TIMESTAMPTZ,
    last_activity_date   TIMESTAMPTZ,
    link                 TEXT,
    is_answered          BOOLEAN,
    raw_json              JSONB NOT NULL,
    inserted_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ------------------------------------------------------------
-- Answers
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.answers (
    answer_id            BIGINT PRIMARY KEY,
    question_id          BIGINT NOT NULL,
    body                 TEXT,
    score                INTEGER,
    is_accepted          BOOLEAN,
    owner_display_name   TEXT,
    creation_date        TIMESTAMPTZ,
    last_activity_date   TIMESTAMPTZ,
    link                 TEXT,
    raw_json             JSONB NOT NULL,
    inserted_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_answers_question
        FOREIGN KEY (question_id)
        REFERENCES public.questions(question_id)
        ON DELETE CASCADE
);


-- ------------------------------------------------------------
-- Ingestion Runs
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.ingestion_runs (
    run_id               UUID PRIMARY KEY,
    started_at            TIMESTAMPTZ NOT NULL,
    completed_at          TIMESTAMPTZ,
    status                VARCHAR(20) NOT NULL,
    questions_extracted  INTEGER NOT NULL DEFAULT 0,
    answers_extracted    INTEGER NOT NULL DEFAULT 0,
    error_message        TEXT,

    CONSTRAINT ingestion_runs_status_check
        CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED'))
);


-- ------------------------------------------------------------
-- Indexes
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_questions_tags
    ON public.questions USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_questions_creation_date
    ON public.questions(creation_date);

CREATE INDEX IF NOT EXISTS idx_questions_last_activity
    ON public.questions(last_activity_date);

CREATE INDEX IF NOT EXISTS idx_answers_question_id
    ON public.answers(question_id);

CREATE INDEX IF NOT EXISTS idx_answers_score
    ON public.answers(score);

CREATE INDEX IF NOT EXISTS idx_answers_creation_date
    ON public.answers(creation_date);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at
    ON public.ingestion_runs(started_at);



-- ============================================================
-- RAG DOCUMENTS
-- One RAG document per Stack Overflow answer
-- ============================================================

CREATE TABLE IF NOT EXISTS public.rag_documents (
    document_id TEXT PRIMARY KEY,

    question_id BIGINT NOT NULL,
    answer_id BIGINT NOT NULL UNIQUE,

    title TEXT NOT NULL,

    question_body TEXT,
    answer_body TEXT NOT NULL,

    tags TEXT[],

    question_score INTEGER,
    answer_score INTEGER,

    is_accepted BOOLEAN NOT NULL DEFAULT FALSE,

    source_url TEXT NOT NULL,

    question_created_at TIMESTAMPTZ,
    answer_created_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_rag_question
        FOREIGN KEY (question_id)
        REFERENCES public.questions(question_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_rag_answer
        FOREIGN KEY (answer_id)
        REFERENCES public.answers(answer_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_question_id
    ON public.rag_documents(question_id);

CREATE INDEX IF NOT EXISTS idx_rag_documents_tags
    ON public.rag_documents USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_rag_documents_answer_score
    ON public.rag_documents(answer_score);

CREATE INDEX IF NOT EXISTS idx_rag_documents_is_accepted
    ON public.rag_documents(is_accepted);
























