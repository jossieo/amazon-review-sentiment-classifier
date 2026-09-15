#!/usr/bin/env python3
"""Cross-check the numbers quoted in README.md against the saved output files."""
import csv, json, re, gzip
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
errors = []
def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond: errors.append(msg)
def has(s): return s in (HERE/"README.md").read_text(encoding="utf-8")

# whole file
tot = 0; g = Counter()
with gzip.open(HERE/"Gift_Cards.jsonl.gz","rt",encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rt = json.loads(line)["rating"]; tot += 1
        g["POS" if rt>=4 else ("NEU" if rt==3 else "NEG")] += 1
check(tot == 152410, f"whole file = 152,410 (got {tot})")
for c, n in [("POS",134940),("NEU",3271),("NEG",14199)]:
    check(g[c]==n, f"class {c} = {n} (got {g[c]})")
check(has("152,410") and has("134,940") and has("3,271") and has("14,199"),
      "README quotes whole-file counts")

# Run 1 binary
b = list(csv.DictReader(open(HERE/"predictions.csv",encoding="utf-8")))
correct = sum(r["correct"]=="True" for r in b)
check(len(b)==100 and correct==98, f"binary: 98/100 (got {correct}/{len(b)})")
pos = sum(r["truth"]=="POS" for r in b); neg = sum(r["truth"]=="NEG" for r in b)
check(pos==93 and neg==7, f"binary classes 93/7 (got {pos}/{neg})")
check(has("98.0%") and has("98/100") and has("93 positive, 7 negative") or has("93 positive"), "README quotes binary headline")

# emotions (Step 5 comparison)
cm = list(csv.DictReader(open(HERE/"emotions_comparison.csv",encoding="utf-8")))
agree_both_defined = 0; defined = 0; joy_vs_ant = 0
nrc_ant = nrc_joy = nrc_none = llm_joy = 0
row5 = None
for r in cm:
    le, ne = r["llm_emotion"], r["nrc_emotion"]
    if le in {"anger","anticipation","disgust","fear","joy","sadness","surprise","trust"} and ne in {"anger","anticipation","disgust","fear","joy","sadness","surprise","trust"}:
        defined += 1
        if le==ne: agree_both_defined += 1
        if le=="joy" and ne=="anticipation": joy_vs_ant += 1
    if ne=="anticipation": nrc_ant+=1
    if ne=="joy": nrc_joy+=1
    if ne=="none": nrc_none+=1
    if le=="joy": llm_joy+=1
    if r["idx"]=="4": row5=(le, ne)
check(nrc_ant==59, f"NRC anticipation=59 (got {nrc_ant})"); check(nrc_joy==21, f"NRC joy=21 (got {nrc_joy})")
check(nrc_none==15, f"NRC none=15 (got {nrc_none})"); check(llm_joy==87, f"LLM joy=87 (got {llm_joy})")
check(agree_both_defined==21 and defined==85, f"emotion agree 21/85 (got {agree_both_defined}/{defined})")
check(joy_vs_ant==51, f"joy-vs-anticipation=51 (got {joy_vs_ant})")
check(row5==("anger","joy"), f"row5 LLM=anger NRC=joy (got {row5})")

# balanced run
ba = list(csv.DictReader(open(HERE/"predictions_balanced.csv",encoding="utf-8")))
bcorrect = sum(r["truth"]==r["llm_sent"] for r in ba)
check(len(ba)==150 and bcorrect==112, f"balanced 112/150 (got {bcorrect}/{len(ba)})")
cm3 = {t: Counter(r["llm_sent"] for r in ba if r["truth"]==t) for t in ["POS","NEU","NEG"]}
expect = {"POS":{"POS":48,"NEU":1,"NEG":1},"NEU":{"POS":5,"NEU":16,"NEG":29},"NEG":{"POS":1,"NEU":1,"NEG":48}}
for t in ["POS","NEU","NEG"]:
    check(dict(cm3[t])==expect[t], f"confusion row {t} = {expect[t]} (got {dict(cm3[t])})")
check(has("74.7%") and has("0.715") and has("33.3%") and has("0.471") and has("32.0%"),
      "README quotes balanced headline + NEU numbers")
check(has("29 (58%)"), "README quotes NEU->NEG 29 (58%)")

print("\n" + ("ALL README NUMBER CHECKS PASSED" if not errors else f"{len(errors)} CHECK(S) FAILED"))
raise SystemExit(1 if errors else 0)
