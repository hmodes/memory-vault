-- Memory Vault — normalise path/version separators before tsvector lexing
--
-- The full-text arm indexed content verbatim, but the Postgres parser keeps
-- slash/dot/dash/tilde sequences together as URL-like lexemes:
-- "ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL" became the single lexeme
-- "opus/sonnet/haiku_model", "v1.18.28" stayed one token, and "memory-vault"
-- never produced a standalone "vault".
--
-- The query side (_build_tsquery in services/search.py) splits on every
-- non-alphanumeric character and ANDs the resulting words, so an
-- exact-identifier query could never match: the AND required plain lexemes
-- ("opus", "18", "vault") that no chunk contained. Chunks holding the
-- identifier verbatim were invisible to keyword search even though the
-- hybrid design promises full-text recall as the complement to vectors.
--
-- Fix: replace the separators the query tokenizer splits on ( / . - ~ )
-- with spaces before lexing, so both sides agree about word boundaries.
-- Underscores need no help — the parser already splits on them on both
-- sides. Content itself is untouched; this only changes the FTS index.
--
-- The trigger change covers new writes; the UPDATE re-lexes every existing
-- row with the same expression.

CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector(
        'english',
        regexp_replace(COALESCE(NEW.content, ''), '[/.~-]+', ' ', 'g')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

UPDATE chunks
SET content_tsv = to_tsvector(
    'english',
    regexp_replace(COALESCE(content, ''), '[/.~-]+', ' ', 'g')
);
