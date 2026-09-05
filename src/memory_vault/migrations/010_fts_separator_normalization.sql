-- Memory Vault — normalise separators before tsvector lexing
--
-- The full-text arm indexed content verbatim, and the Postgres parser keeps
-- punctuation-bearing sequences together as URL/host/email-like lexemes:
-- "ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL" produced "opus/sonnet/haiku_model"
-- as one lexeme next to standalone words, "1.18.28" stayed one token, and
-- "hmodes@example.com" never produced "example". (Hyphenated words are NOT
-- part of the motivation: the parser already emits both the compound and its
-- parts. Normalising them here is for index consistency, not recall.)
--
-- The query side (_build_tsquery in services/search.py) splits on every
-- non-alphanumeric character and ANDs the resulting words, so an
-- exact-identifier query could never match: the AND required plain lexemes
-- ("opus", "18", "example") that no chunk contained. Chunks holding the
-- identifier verbatim were invisible to keyword search even though the
-- hybrid design promises full-text recall as the complement to vectors.
--
-- Fix: replace every non-alphanumeric run with a space before lexing — the
-- same effective rule the query tokenizer applies to ASCII input. The POSIX
-- class [^[:alnum:]] is locale-aware when the database ctype is (the common
-- case, e.g. *.utf8 initdb), so accented and non-Latin words stay intact;
-- under a C ctype it degrades to ASCII, matching the old parser behaviour.
-- Content itself is untouched; this only changes the FTS index.
--
-- Trade-offs, deliberately accepted:
--   * Word-level matching is not phrase matching: the query side ANDs the
--     fragments of a compound with no adjacency requirement, so "1.18.28"
--     matches any chunk mentioning 1, 18 and 28 separately. That is the
--     existing _build_tsquery semantics; this change makes such queries
--     match at all instead of never.
--   * Compound lexemes (emails, URLs, IPs as single tokens) disappear from
--     the index. Every reader in this codebase goes through _build_tsquery
--     (verified: no plainto/websearch/phraseto_tsquery or ts_headline use),
--     which could never use them anyway.
--   * Pathologically punctuation-dense content near remember()'s size cap
--     can exceed the 1 MB tsvector limit once shredded into short tokens,
--     where the old parser skipped >2047-byte tokens. The file-ingest path
--     is word-capped (500/chunk) and cannot reach it.
--
-- The trigger change covers new writes; the UPDATE re-lexes existing rows
-- with the same expression, skipping rows whose vector is already correct so
-- a re-run (or a large mostly-separator-free table) does not rewrite
-- unchanged rows. The migration is idempotent; a row inserted by a
-- concurrent transaction that cached the old function definition is picked
-- up by a re-run.

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
