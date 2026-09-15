#!/usr/bin/env python3
"""Binary sentiment classifier over the first batch of Gift Card reviews.

- Model sees ONLY review title + text (never the rating).
- Ground truth computed AFTERWARD from rating: >=4 -> POSITIVE, else NEGATIVE.
- Calls an OpenAI-compatible endpoint (base_url/model/api_key read from
  the user's Hermes config.yaml at runtime; nothing secret is hardcoded).
- Writes predictions to CSV and prints a per-class + wrong-row report.
"""
import argparse, csv, json, os, time
from pathlib import Path

def load_endpoint(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {})
    base = m.get("base_url")
    key = m.get("api_key")
    model = m.get("default") or m.get("model")
    if not key:
        for cp in cfg.get("custom_providers", []):
            if cp.get("base_url") == base and cp.get("api_key"):
                key = cp["api_key"]
    return {"base_url": base, "api_key": key, "model": model}

SYSTEM_PROMPT = (
    "You are a binary sentiment classifier for product reviews. "
    "You will be given a review's title and text. Determine the sentiment "
    "conveyed by the review. Reply with exactly ONE word: POSITIVE or NEGATIVE. "
    "Do not include any other text or explanation."
)

def parse_label(raw):
    """Normalize the model's reply to 'POS'/'NEG' or 'UNKNOWN'."""
    if raw is None:
        return "UNKNOWN"
    t = raw.strip().upper().split()[0] if raw.strip() else ""
    t = t.rstrip(".,!?;:\"'\u2019\u201d\u201c")
    if t.startswith("POS"):
        return "POS"
    if t.startswith("NEG"):
        return "NEG"
    return "UNKNOWN"

def classify(client, ep, title, text, api_key):
    user_msg = f"Title: {title.strip()}\n\nReview text: {text.strip()}"
    resp = client.chat.completions.create(
        model=ep["model"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content
    return parse_label(raw), raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", help="path to jsonl of the batch (must contain rating, title, text)")
    ap.add_argument("--endpoint-config", default=str(Path.home()/".hermes/config.yaml"))
    ap.add_argument("--out-csv", default="predictions.csv")
    args = ap.parse_args()

    ep = load_endpoint(args.endpoint_config)
    from openai import OpenAI
    client = OpenAI(base_url=ep["base_url"], api_key=ep["api_key"])

    rows = [json.loads(l) for l in open(args.batch, encoding="utf-8") if l.strip()]
    print(f"classifying {len(rows)} reviews via {ep['model']} @ {ep['base_url']}...\n")

    results = []
    for i, r in enumerate(rows):
        rating = r["rating"]
        truth = "POS" if rating >= 4 else "NEG"
        title = r.get("title") or ""
        text = r.get("text") or ""
        # Retry once on transient API errors.
        for attempt in (1, 2):
            try:
                pred, raw = classify(client, ep, title, text, api_key=ep["api_key"])
                break
            except Exception as e:
                if attempt == 2:
                    pred, raw = "UNKNOWN", f"ERROR: {e}"
                else:
                    time.sleep(1.5)
        results.append({
            "idx": i,
            "asin": r.get("asin", ""),
            "rating": rating,
            "truth": truth,
            "pred": pred,
            "correct": pred == truth,
            "raw": (raw or "")[:400],
            "title": title[:120],
        })
        status = "ok" if pred != "UNKNOWN" else "UNKNOWN"
        print(f"  [{i+1:3d}] truth={truth:3s} pred={pred:3s} {status} | {title[:70]!r}")

    # ---- scoring ----
    def metrics(predset_true, predset_hat):
        tp = sum(p == "POS" and t == "POS" for p, t in zip(predset_hat, predset_true))
        fp = sum(p == "POS" and t == "NEG" for p, t in zip(predset_hat, predset_true))
        fn = sum(p == "NEG" and t == "POS" for p, t in zip(predset_hat, predset_true))
        tn = sum(p == "NEG" and t == "NEG" for p, t in zip(predset_hat, predset_true))
        prec_pos = tp / (tp + fp) if tp + fp else float("nan")
        rec_pos  = tp / (tp + fn) if tp + fn else float("nan")
        prec_neg = tn / (tn + fn) if tn + fn else float("nan")
        rec_neg  = tn / (tn + fp) if tn + fp else float("nan")
        f1 = lambda p, r: (2 * p * r / (p + r)) if (p + r) else float("nan")
        return {
            "n": len(predset_true),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pos": {"n": tp + fn, "prec": prec_pos, "rec": rec_pos, "f1": f1(prec_pos, rec_pos)},
            "neg": {"n": tn + fp, "prec": prec_neg, "rec": rec_neg, "f1": f1(prec_neg, rec_neg)},
        }

    # Treat UNKNOWN predictions as a disagreement (count as incorrect / not NEG nor POS).
    defined = [r for r in results if r["pred"] != "UNKNOWN"]
    ndef = len(defined)
    acc = sum(r["correct"] for r in results) / len(results)
    disp = sum(r["pred"] != r["truth"] for r in results)
    m_defined = metrics([r["truth"] for r in defined], [r["pred"] for r in defined])
    m_all = metrics([r["truth"] for r in results], [("POS" if r["pred"]=="POS" else "NEG") for r in results])

    # ---- report ----
    lines = []
    lines.append("="*70)
    lines.append("SCORING REPORT — first batch vs. ground truth (rating>=4 -> POS)")
    lines.append("="*70)
    lines.append(f"Reviews in batch            : {len(results)}")
    lines.append(f"Ground truth: POSITIVE      : {sum(r['truth']=='POS' for r in results)}  ({(sum(r['truth']=='POS' for r in results)/len(results))*100:.1f}%)")
    lines.append(f"              NEGATIVE      : {sum(r['truth']=='NEG' for r in results)}  ({(sum(r['truth']=='NEG' for r in results)/len(results))*100:.1f}%)")
    lines.append(f"Undefined/UNKNOWN preds     : {len(results)-ndef}")
    lines.append(f"Model agrees with rating    : {sum(r['correct'] for r in results)}/{len(results)} = {acc*100:.1f}%  (overall)")
    lines.append(f"Disagreements (model wrong) : {disp}")
    lines.append("")
    lines.append("Contingency (predict vs truth), UNKNOWN=treated as incorrect:")
    lines.append(f"                     truth=POS  truth=NEG")
    lines.append(f"  predict=POS            {m_all['tp']:3d}         {m_all['fp']:3d}")
    lines.append(f"  predict=NEG            {m_all['fn']:3d}         {m_all['tn']:3d}")
    lines.append("")
    lines.append("Per-class agreement (defined predictions only; UNKNOWN excluded):")
    lines.append(f"  POSITIVE class  n={m_defined['pos']['n']:3d}  precision={m_defined['pos']['prec']*100:5.1f}%  recall={m_defined['pos']['rec']*100:5.1f}%  F1={m_defined['pos']['f1']:0.3f}")
    lines.append(f"  NEGATIVE class  n={m_defined['neg']['n']:3d}  precision={m_defined['neg']['prec']*100:5.1f}%  recall={m_defined['neg']['rec']*100:5.1f}%  F1={m_defined['neg']['f1']:0.3f}")
    if ndef:
        lines.append(f"  (macro F1 over the {ndef} defined predictions: "
                     f"{(m_defined['pos']['f1']+m_defined['neg']['f1'])/2:0.3f})")
    lines.append("")
    lines.append("REVIEWS THE MODEL GOT WRONG (incl. UNKNOWN):")
    wrong = [r for r in results if not r["correct"]]
    if not wrong:
        lines.append("  none")
    for r in wrong:
        lines.append(f"  row#{r['idx']}  rating={r['rating']} (truth={r['truth']})  model={r['pred']}")
        lines.append(f"       title: {r['title']!r}")
        if r["raw"] and not r["raw"].startswith("ERROR") and r["pred"] != "UNKNOWN":
            lines.append(f"       model raw: {r['raw'][:160]!r}")
    lines.append("")
    lines.append("NOTE: model never saw the rating; ground truth derived only after inference.")
    report = "\n".join(lines)
    print(report)

    # save CSV + report
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "asin", "rating", "truth", "pred", "correct", "title", "raw"])
        for r in results:
            w.writerow([r["idx"], r["asin"], r["rating"], r["truth"], r["pred"],
                        r["correct"], r["title"], r["raw"]])
    with open("score_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[saved] {args.out_csv} and score_report.txt")

if __name__ == "__main__":
    main()
