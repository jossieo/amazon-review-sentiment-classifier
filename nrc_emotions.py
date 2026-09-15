#!/usr/bin/env python3
"""Word-list take on Step 5: primary-emotion detection via the NRC Emotion Lexicon.

Reads a batch of reviews, scores each review's words (title + text, lowercased)
against the NRC Word-Emotion Association Lexicon (EmoLex, v0.92) for the eight
emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.
For each review: sum emotion indicators per emotion; the highest-scoring emotion
is the predicted primary emotion. Ties are broken by the canonical emotion order
above. Reviews with no emotion words score 0 everywhere -> primary = "none".
No model calls; fully deterministic. Saves emotions_nrc.csv
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

def load_lexicon(path):
    """Return {word: set(emotions)} using only association=1 rows for the 8 emotions."""
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emo, val = parts
            if emo not in EMOTIONS or val != "1":
                continue
            lex.setdefault(word.lower(), set()).add(emo)
    return lex

WORD_RE = re.compile(r"[a-z']+")

def tokenize(title, text):
    toks = WORD_RE.findall((title + " " + text).lower())
    return [t.strip("'") for t in toks if t.strip("'")]

def primary_emotion(scores):
    """Argmax over canonical emotion order; 'none' if all zero."""
    best, bestc = "none", 0
    for e in EMOTIONS:
        if scores[e] > bestc:
            best, bestc = e, scores[e]
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", default="data/first_100.jsonl", nargs="?")
    ap.add_argument("--lexicon", default="data/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    ap.add_argument("--out", default="emotions_nrc.csv")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    print(f"lexicon: {len(lex)} words mapped to the 8 NRC emotions")

    rows = [json.loads(l) for l in open(args.batch, encoding="utf-8") if l.strip()]
    out = []
    emo_dist = Counter()
    none = 0
    for i, r in enumerate(rows):
        toks = tokenize(r.get("title") or "", r.get("text") or "")
        scores = {e: 0 for e in EMOTIONS}
        matched = 0
        for t in toks:
            emos = lex.get(t)
            if not emos:
                continue
            matched += 1
            for e in emos:
                scores[e] += 1
        primary = primary_emotion(scores)
        if primary == "none":
            none += 1
        else:
            emo_dist[primary] += 1
        out.append({"idx": i, "rating": r["rating"],
                    "tokens": len(toks), "matched_tokens": matched,
                    "nrc_emotion": primary,
                    **{f"nrc_{e}": scores[e] for e in EMOTIONS},
                    "title": (r.get("title") or "").strip()[:120]})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"saved {args.out}  ({len(out)} reviews)")
    print(f"no emotion words found (primary=none): {none}")
    print("NRC primary-emotion distribution:")
    for e in EMOTIONS:
        print(f"  {e:14s} {emo_dist.get(e, 0)}")

if __name__ == "__main__":
    main()
