#!/usr/bin/env python3
"""Step 6 — three-class LLM scoring over a balanced sample.

Redefines sentiment to POSITIVE / NEUTRAL / NEGATIVE (3-star = NEUTRAL) and keeps
the Step-5 primary emotion. Output format one line: SENTIMENT|EMOTION.
Model sees only title + text. Writes predictions_balanced.csv
"""
import csv, json, re, time
from pathlib import Path
from collections import Counter

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
SENTIMENTS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

def load_endpoint(config_path):
    import yaml
    cfg = yaml.safe_load(open(config_path))
    m = cfg.get("model", {})
    base, key = m.get("base_url"), m.get("api_key")
    model = m.get("default") or m.get("model")
    if not key:
        for cp in cfg.get("custom_providers", []):
            if cp.get("base_url") == base and cp.get("api_key"):
                key = cp["api_key"]
    return {"base_url": base, "api_key": key, "model": model}

SYSTEM = (
    "You are a sentiment and emotion classifier for product reviews.\n"
    "Given a review's title and text, output the overall sentiment as one of "
    "POSITIVE, NEUTRAL, or NEGATIVE, and the single primary emotion.\n"
    "NEUTRAL means the review is neither clearly positive nor clearly negative "
    "(mixed, lukewarm, 3-star-style, or factual).\n"
    "Reply with exactly ONE line, this format, and nothing else:\n"
    "SENTIMENT|EMOTION\n"
    "SENTIMENT is POSITIVE, NEUTRAL, or NEGATIVE. EMOTION is exactly one of: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust (lowercase)."
)

def parse(raw):
    if not raw:
        return "UNKNOWN", "UNKNOWN", False
    m = re.search(r"(POSITIVE|NEUTRAL|NEGATIVE)\s*\|\s*([a-z]+)", raw, re.IGNORECASE)
    if not m:
        return "UNKNOWN", "UNKNOWN", False
    sent = m.group(1).upper()
    emo = re.sub(r"[^a-z]", "", m.group(2).lower())
    return sent, emo, True

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", default="data/balanced_150.jsonl", nargs="?")
    ap.add_argument("--endpoint-config", default=str(Path.home() / ".hermes/config.yaml"))
    ap.add_argument("--out", default="predictions_balanced.csv")
    args = ap.parse_args()

    ep = load_endpoint(args.endpoint_config)
    from openai import OpenAI
    client = OpenAI(base_url=ep["base_url"], api_key=ep["api_key"])

    rows = [json.loads(l) for l in open(args.batch, encoding="utf-8") if l.strip()]
    print(f"classifying {len(rows)} reviews (3-class) via {ep['model']}…", flush=True)

    out = []
    for i, r in enumerate(rows):
        title, text = (r.get("title") or "").strip(), (r.get("text") or "").strip()
        user = f"Title: {title}\n\nReview text: {text}"
        raw = ""
        for attempt in (1, 2):
            try:
                resp = client.chat.completions.create(
                    model=ep["model"], temperature=0,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                )
                raw = resp.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt == 2:
                    raw = f"ERROR: {e}"
                else:
                    time.sleep(1.5)
                continue
        sent, emo, ok = parse(raw)
        truth = "POS" if r["rating"] >= 4 else ("NEU" if r["rating"] == 3 else "NEG")
        # map full-word sentiment -> short class for bookkeeping
        sent_cls = {"POSITIVE": "POS", "NEUTRAL": "NEU", "NEGATIVE": "NEG"}.get(sent, "UNK")
        out.append({"idx": i, "asin": r.get("asin", ""), "rating": r["rating"],
                    "truth": truth, "llm_sent_full": sent, "llm_sent": sent_cls,
                    "llm_emotion": emo, "parse_ok": ok, "title": title[:120]})
        print(f"  [{i+1:3d}] truth={truth}  pred={sent_cls:3s}  emotion={emo:12s} | {title[:55]!r}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nsaved {args.out}", flush=True)
    fails = sum(1 for o in out if not o["parse_ok"])
    print(f"parse failures: {fails}", flush=True)
    print("predicted-class distribution:", dict(Counter(o["llm_sent"] for o in out)), flush=True)

if __name__ == "__main__":
    main()
