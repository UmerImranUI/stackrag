


-- ============================================================
-- BUILD RAG DOCUMENTS
-- One document per answer
-- ============================================================

TRUNCATE TABLE public.rag_documents;

INSERT INTO public.rag_documents (
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
    question_created_at,
    answer_created_at
)
SELECT
    'stackoverflow_answer_' || a.answer_id::TEXT AS document_id,

    q.question_id,
    a.answer_id,

    q.title,

    q.body AS question_body,
    a.body AS answer_body,

    q.tags,

    q.score AS question_score,
    a.score AS answer_score,

    COALESCE(a.is_accepted, FALSE) AS is_accepted,

    COALESCE(a.link, q.link) AS source_url,

    q.creation_date AS question_created_at,
    a.creation_date AS answer_created_at

FROM public.answers a
INNER JOIN public.questions q
    ON a.question_id = q.question_id

WHERE a.body IS NOT NULL
  AND LENGTH(TRIM(a.body)) > 0;