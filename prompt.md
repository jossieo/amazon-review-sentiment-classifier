# Prompt — Amazon review sentiment + primary emotion

This is the LLM prompt used to score each review. It is **reusable** (lives in
`score_balanced.py` / `score_llm_emotions.py` as `SYSTEM`), takes only a review's
`title` and `text`, never the `rating`, and forces a single parseable line so the
output can be read back programmatically.

## Final version (3-class sentiment + emotion, used for the balanced run)

> You are a sentiment and emotion classifier for product reviews.
> Given a review's title and text, output the overall sentiment as one of
> POSITIVE, NEUTRAL, or NEGATIVE, and the single primary emotion.
> NEUTRAL means the review is neither clearly positive nor clearly negative
> (mixed, lukewarm, 3-star-style, or factual).
> Reply with exactly ONE line, this format, and nothing else:
> `SENTIMENT|EMOTION`
> SENTIMENT is POSITIVE, NEUTRAL, or NEGATIVE. EMOTION is exactly one of:
> anger, anticipation, disgust, fear, joy, sadness, surprise, trust (lowercase).

### Parsing contract

The model replies with exactly one line `SENTIMENT|EMOTION` (regex:
`(POSITIVE|NEUTRAL|NEGATIVE)\s*\|\s*([a-z]+)`). Temperature is set to `0` for
determinism. Unparseable replies are flagged, counted, and treated as errors —
never silently assumed.

## Earlier version (Step 2, binary sentiment)

Same structure but sentiment is limited to `POSITIVE | NEGATIVE` (3-star folded
into NEGATIVE per the Step-2 rule), reproduced in `score_batch.py`.

## Edge cases decided (per Step 1)

- **Conflicting title vs. text** — the model reads both; the review's text is the
  stronger signal (e.g., a 5-star review whose *text* complains → the words win).
- **Terse / angry short reviews** — handled by the LLM's reading; the word-list
  take has no such ability (documented divergence in the report).
- **Emoji / punctuation-only reviews** — the LLM usually infers the emotion; the
  NRC word list scores "none" when no lexicon word is present.
- **Output shape** — a single `SENTIMENT|EMOTION` line is deliberately
  machine-readable; "POSITIVE"/"NEGATIVE"/"NEUTRAL" and the 8 emotion names are
  the only valid tokens.
