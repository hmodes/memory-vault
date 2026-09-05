-- Memory Vault — normalise separators before tsvector lexing
--
-- The full-text arm indexed content verbatim, but the Postgres parser keeps
-- punctuation-bearing sequences together as URL/host/email-like lexemes:
-- "ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL" produced "opus/sonnet/haiku_model"
-- as one lexeme next to standalone words, "1.18.28" stayed one token, and
-- "memory-vault" never produced a standalone "vault".
--
-- The query side (_build_tsquery in services/search.py) splits on every
-- non-alphanumeric character and ANDs the resulting words, so an
-- exact-identifier query could never match: the AND required plain lexemes
-- ("opus", "18", "vault") that no chunk contained. Chunks holding the
-- identifier verbatim were invisible to keyword search even though the
-- hybrid design promises full-text recall as the complement to vectors.
--
-- Fix: replace every non-alphanumeric run with a space before lexing — the
-- same effective rule the query tokenizer applies to ASCII input. The POSIX
-- class [[:alnum:]] is locale-aware, so accented and non-Latin words stay
-- intact; only punctuation splits. Content itself is untouched; this only
-- changes the FTS index.
--
-- The trigger change covers new writes; the UPDATE re-lexes existing rows
-- with the same expression, skipping rows whose vector is already correct so
-- a re-run (or a large table where most rows lack separators) does not
-- rewrite unchanged rows.

CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector(
        'english',
        regexp_replace(COALESCE(NEW.content, ''), '[^[:alnum:]]+', ' ', 'g')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

UPDATE chunks
SET content_tsv = to_tsvector(
    'english',
    regexp_replace(COALESCE(content, ''), '[^[:alnum:]]+', ' ', 'g')
)
WHERE content_tsv IS DISTINCT FROM to_tsvector(
    'english',
    regexp_replace(COALESCE(content, ''), '[^[:alnum:]]+', ' ', 'g')
);
