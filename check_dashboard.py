#!/usr/bin/env python3
"""Verify dashboard.html against the SAVED output (the brief's in-browser re-check).

Checks the embedded review payload and headline figures against the CSVs/JSONL the
page was generated from, then cross-checks the saved confusion_balanced.csv.
"""
import csv, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
html = (HERE / "dashboard.html").read_text(encoding="utf-8")

errors = []
def check(cond, msg):
    print(("  OK  " if cond else "  FAIL ") + msg)
    if not cond: errors.append(msg)

# 1) embedded rows must match predictions_balanced.csv exactly
m = re.search(r'<script type="application/json" id="rows-data">(.*?)</script>', html, re.S)
rows = json.loads(m.group(1))
print("\n-- embedded payload vs saved predictions_balanced.csv --")
saved = {int(r["idx"]): r for r in csv.DictReader(open(HERE/"predictions_balanced.csv", encoding="utf-8"))}
check(len(rows) == 150, f"payload has {len(rows)} reviews (expect 150)")
mism = 0
for r in rows:
    s = saved[r["idx"]]
    truth_ok = r["truth"] == s["truth"]
    pred_ok = r["pred"] == s["llm_sent"]
    if not (truth_ok and pred_ok): mism += 1
check(mism == 0, f"every embedded truth/pred matches saved CSV (mismatches={mism})")
by_truth = {"POS": 0, "NEU": 0, "NEG": 0}
for r in rows: by_truth[r["truth"]] += 1
check(by_truth == {"POS": 50, "NEU": 50, "NEG": 50}, f"payload balanced 50/50/50 -> {by_truth}")
correct = sum(r["correct"] for r in rows)
check(correct == 112, f"payload agrees-with-rating count = {correct} (expect 112)")

# 2) key headline figures present in the HTML
def needle(s):
    return html.count(s)
print("\n-- headline figures present in the page --")
for tok, n in [("74.7%",2), ("0.715",1), ("88.5%",1), ("2.1%",1), ("9.3%",1),
               ("33.3%",1), ("112 of 150",0), ("38",0)]:
    c = needle(tok)
    ok = c >= (1 if n == 0 else n)
    check(ok, f"'{tok}' x{c}" + ("" if ok else "  <-- missing/low"))
# confusion matrix numbers
for num in ["48", "16", "29"]:
    check(html.count(f">{num}<")>=1 or html.count(num) >= 1, f"confusion count present: {num}")

# 3) cross-check against saved confusion_balanced.csv
print("\n-- against confusion_balanced.csv --")
cm_saved = {}
for r in csv.DictReader(open(HERE/"confusion_balanced.csv", encoding="utf-8")):
    cm_saved[r["truth"]] = {k: int(v) for k, v in r.items() if k != "truth"}
from collections import Counter
cm_from_payload = {t: Counter(r["pred"] for r in rows if r["truth"] == t) for t in ["POS","NEU","NEG"]}
for t in ["POS","NEU","NEG"]:
    check({k: cm_from_payload[t][k] for k in cm_saved[t]} == cm_saved[t], f"row {t}: payload {dict(cm_from_payload[t])} == saved {cm_saved[t]}")

# 4) star distribution sanity from the file
print("\n-- star distribution sanity --")
from collections import Counter as C2
stars = C2()
for line in (HERE/"data/balanced_150.jsonl").read_text(encoding="utf-8").splitlines():
    stars[int(json.loads(line)["rating"])] += 1
check(sum(stars.values()) == 150 and stars[3] == 50, f"stars {dict(sorted(stars.items()))} (3-star=50)")

print("\n" + ("ALL CHECKS PASSED" if not errors else f"{len(errors)} CHECK(S) FAILED"))
sys.exit(1 if errors else 0)
