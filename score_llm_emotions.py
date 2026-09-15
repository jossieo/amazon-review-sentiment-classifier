#!/usr/bin/env python3
"""LLM take on Step 5: primary-emotion detection.

Extends the Step-1 prompt so each review yields BOTH a binary sentiment AND a
single primary emotion. Output format is one line:  SENTIMENT|EMOTION
  SENTIMENT in {POSITIVE, NEGATIVE}
  EMOTION  in {anger, anticipation, disgust, fear, joy, sadness, surprise, trust}
The model sees only title + text (never the rating). Saves predictions_with_emotions.csv
"""
import argparse, csv, json, re, time
from pathlib import Path

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

def load_endpoint(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {})
    base = m.get("base_url")
    key = m.get("api_key")
    model = m.get("default") or m.get("model")
    # custom provider may hold the key when provider == "custom"
    if not key:
        for cp in cfg.get("custom_providers", []):
            if cp.get("base_url") == base and cp.get("api_key"):
                key = cp["api_key"]
    return {"base_url": base, "api_key": key, "model": model}

SYSTEM = (
    "You are a sentiment and emotion classifier for product reviews.\n"
    "Given a review's title and text, output the overall sentiment and the single "
    "primary emotion the reviewer conveys.\n"
    "Reply with exactly ONE line, this format, and nothing else:\n"
    "SENTIMENT|EMOTION\n"
    "SENTIMENT is POSITIVE or NEGATIVE. EMOTION is exactly one of: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust (lowercase)."
)

def parse(raw):
    """Return (sentiment, emotion, ok)."""
    if not raw:
        return "UNKNOWN", "UNKNOWN", False
    m = re.search(r"(POSITIVE|NEGATIVE)\s*\|\s*([a-z]+)", raw, re.IGNORECASE)
    if not m:
        return "UNKNOWN", "UNKNOWN", False
    sent = m.group(1).upper()
    emo = m.group(2).lower()
    emo = re.sub(r"[^a-z]", "", emo)
    return sent, emo, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", default="data/first_100.jsonl", nargs="?")
    ap.add_argument("--endpoint-config", default=str(Path.home() / ".hermes/config.yaml"))
    ap.add_argument("--out", default="predictions_with_emotions.csv")
    args = ap.parse_args()

    ep = load_endpoint(args.endpoint_config)
    from openai import OpenAI
    client = OpenAI(base_url=ep["base_url"], api_key=ep["api_key"])

    rows = [json.loads(l) for l in open(args.batch, encoding="utf-8") if l.strip()]
    print(f"classifying {len(rows)} reviews via {ep['model']}…\n")

    out = []
    for i, r in enumerate(rows):
        title, text = (r.get("title") or "").strip(), (r.get("text") or "").strip()
        user = f"Title: {title}\n\nReview text: {text}"
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
        out.append({"idx": i, "rating": r["rating"], "truth": "POS" if r["rating"] >= 4 else "NEG",
                    "llm_sent": sent, "llm_emotion": emo, "parse_ok": ok,
                    "title": title[:120], "asin": r.get("asin", "")})
        flag = "" if ok else "  <-- PARSE FAIL"
        print(f"  [{i+1:3d}] sent={sent:8s} emotion={emo:14s}{flag} | {title[:60]!r}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    fails = sum(1 for o in out if not o["parse_ok"])
    from collections import Counter
    emo_dist = Counter(o["llm_emotion"] for o in out)
    print(f"\nsaved {args.out}  |  parse failures: {fails}")
    print("LLM primary-emotion distribution:")
    for e in EMOTIONS:
        print(f"  {e:14s} {emo_dist.get(e, 0)}")
    other = {k: v for k, v in emo_dist.items() if k not in EMOTIONS}
    if other:
        print("  outside the 8-class set:", dict(other))

if __name__ == "__main__":
    main()
