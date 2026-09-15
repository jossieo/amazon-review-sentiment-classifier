#!/usr/bin/env python3
"""Step 6 — three-class evaluation of the balanced run.

Computes the 3x3 confusion matrix (pred vs truth), per-class precision / recall / F1,
overall accuracy, macro-F1, and a baseline. Focuses on the NEUTRAL question: do 3-star
reviews survive as their own class or collapse, and in which direction? Reads
predictions_balanced.csv; pulls review text from the balanced sample for examples.
Writes eval_balanced_report.txt and confusion_balanced.csv
"""
import csv, json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["POS", "NEU", "NEG"]

def load_texts():
    out = {}
    for line in (HERE / "data/balanced_150.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r.get("idx", len(out))] = (r.get("title") or "", (r.get("text") or "")[:300])
    return out

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0

def main():
    texts = load_texts()
    rows = list(csv.DictReader(open(HERE / "predictions_balanced.csv", encoding="utf-8")))
    assert len(rows) == 150, len(rows)

    unk = sum(1 for r in rows if r["llm_sent"] not in CLASSES)
    # confusion matrix: rows truth, cols predicted
    cm = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for r in rows:
        p = r["llm_sent"] if r["llm_sent"] in CLASSES else "UNK"
        if p in CLASSES:
            cm[r["truth"]][p] += 1
    correct = sum(cm[t][t] for t in CLASSES)
    total = len(rows)
    acc = correct / total

    per = {}
    for c in CLASSES:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in CLASSES if t != c)
        fn = sum(cm[c][p] for p in CLASSES if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per[c] = {"n": tp + fn, "prec": prec, "rec": rec, "f1": f1(prec, rec)}
    macro = sum(per[c]["f1"] for c in CLASSES) / len(CLASSES)
    from collections import Counter as C2
    true_n = C2(r["truth"] for r in rows)
    baseline = max(true_n.values()) / total   # always-predict-most-frequent-in-sample

    L = []
    L.append("=" * 62)
    L.append("STEP 6 — THREE-CLASS BALANCED RUN (150 reviews; 50 per class, seed=42)")
    L.append("=" * 62)
    L.append(f"Truth counts in sample     : POS {true_n['POS']}  NEU {true_n['NEU']}  NEG {true_n['NEG']}")
    L.append(f"Unparseable predictions    : {unk}")
    L.append("")
    L.append("Confusion matrix  (rows = truth, cols = model's call):")
    L.append("truth \\ pred  " + "".join(f"{c:>9s}" for c in CLASSES))
    for t in CLASSES:
        L.append(f"{t:10s}" + "".join(f"{cm[t][c]:>8d}" for c in CLASSES))
    L.append("")
    L.append("Overall accuracy           : {:.1f}%  ({}/{})".format(acc * 100, correct, total))
    L.append("Macro F1 (mean of 3)       : {:.3f}".format(macro))
    L.append("Majority baseline (in run) : {:.1f}%".format(baseline * 100))
    L.append("")
    L.append("Per-class (precision / recall / F1):")
    for c in CLASSES:
        pc = per[c]
        L.append(f"  {c:4s} n={pc['n']:3d}  precision={pc['prec']*100:5.1f}%  recall={pc['rec']*100:5.1f}%  F1={pc['f1']:.3f}")
    L.append("")
    # ---- the NEUTRAL question ----
    L.append("THE NEUTRAL (3-star) QUESTION:")
    neu = cm["NEU"]
    L.append(f"  True-NEUTRAL reviews collapsed into POSITIVE: {neu['POS']}")
    L.append(f"  True-NEUTRAL reviews collapsed into NEGATIVE: {neu['NEG']}")
    L.append(f"  True-NEUTRAL reviews kept as NEUTRAL         : {neu['NEU']}")
    # direction of each confusion
    pn = [("POS->NEU", cm["POS"]["NEU"]), ("POS->NEG", cm["POS"]["NEG"]),
          ("NEU->POS", cm["NEU"]["POS"]), ("NEU->NEG", cm["NEU"]["NEG"]),
          ("NEG->NEU", cm["NEG"]["NEU"]), ("NEG->POS", cm["NEG"]["POS"])]
    pn = sorted(pn, key=lambda x: x[1], reverse=True)
    L.append("  Largest misclassifications:")
    for pair, c in pn[:4]:
        if c:
            L.append(f"    {pair:10s} x {c}")
    L.append("")
    L.append("Example 3-star reviews — did the model keep them NEUTRAL?")
    shown = 0
    for r in rows:
        if r["truth"] != "NEU":
            continue
        if shown >= 8:
            break
        ttl = texts.get(int(r["idx"]), ("", ""))[0]
        L.append(f"  #{int(r['idx'])+1:3d} pred={r['llm_sent']:3s} | rating=3  {ttl[:50]!r}")
        shown += 1
    report = "\n".join(L)
    print(report)
    (HERE / "eval_balanced_report.txt").write_text(report, encoding="utf-8")

    with open(HERE / "confusion_balanced.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["truth"] + CLASSES)
        for t in CLASSES:
            w.writerow([t] + [cm[t][c] for c in CLASSES])
    print("\n[saved] eval_balanced_report.txt and confusion_balanced.csv")
    print(f"overall={acc*100:.1f}%  macroF1={macro:.3f}  NEU-kept={cm['NEU']['NEU']}/50  NEU->POS={cm['NEU']['POS']}  NEU->NEG={cm['NEU']['NEG']}")

if __name__ == "__main__":
    main()
