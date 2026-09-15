#!/usr/bin/env python3
"""Compare the two Step-5 primary-emotion takes (LLM vs NRC word list).

Inputs:
  predictions_with_emotions.csv  -> idx, rating, truth, llm_sent, llm_emotion
  emotions_nrc.csv               -> idx, nrc_emotion, per-emotion counts
  data/first_100.jsonl           -> text (for divergence examples)
Outputs: emotions_comparison.csv (per review), emotions_report.txt (summary)
"""
import csv, json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

def load_texts():
    out = {}
    for line in (HERE / "data/first_100.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r.get("idx", len(out))] = (r.get("title") or "", r.get("text") or "")
    return out

def main():
    texts = load_texts()

    llm = {}
    with open(HERE / "predictions_with_emotions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            llm[int(r["idx"])] = r
    nrc = {}
    with open(HERE / "emotions_nrc.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nrc[int(r["idx"])] = r

    idxs = sorted(llm, key=int)
    assert len(idxs) == len(nrc) == 100, (len(idxs), len(nrc))

    agree_any = agree_both = 0
    confused = Counter()          # (llm_emo vs nrc_emo)
    llm_dist, nrc_dist, nrc_none, llm_unknown = Counter(), Counter(), 0, 0
    rows = []
    for i in idxs:
        le = llm[i]["llm_emotion"]; ne = nrc[i]["nrc_emotion"]
        llm_dist[le] += 1; nrc_dist[ne] += 1
        if ne == "none": nrc_none += 1
        if le not in EMOTIONS: llm_unknown += 1
        if le == ne: agree_any += 1
        if le in EMOTIONS and ne in EMOTIONS and le == ne: agree_both += 1
        if le in EMOTIONS and ne in EMOTIONS and le != ne:
            confused[(le, ne)] += 1
        rows.append({"idx": i, "rating": llm[i]["rating"], "title": llm[i]["title"][:120],
                     "llm_sent": llm[i]["llm_sent"], "llm_emotion": le, "nrc_emotion": ne})

    with open(HERE / "emotions_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_defined = sum(1 for r in rows if r["llm_emotion"] in EMOTIONS and r["nrc_emotion"] in EMOTIONS)
    L = []
    L.append("=" * 66)
    L.append("STEP 5 — PRIMARY-EMOTION COMPARISON: LLM vs NRC word list (100 reviews)")
    L.append("=" * 66)
    L.append(f"LLM parse failures/unknown    : {llm_unknown}")
    L.append(f"NRC no-emotion-word reviews  : {nrc_none}")
    L.append(f"Exact agree on emotion (all) : {agree_any}/100 = {agree_any/100:.1%}")
    L.append(f"Exact agree (both defined)   : {agree_both}/{n_defined} = {agree_both/n_defined:.1%}")
    if n_defined:
        L.append(f"  (restricts to the {n_defined} reviews where both predicted a real emotion)")
    L.append("")
    L.append("Emotion distribution (reviews):")
    L.append(f"  {'':14s} {'LLM':>5s} {'NRC':>5s}")
    for e in EMOTIONS:
        L.append(f"  {e:14s} {llm_dist.get(e,0):5d} {nrc_dist.get(e,0):5d}")
    L.append(f"  {'(none/unk)':14s} {llm_unknown:5d} {nrc_none:5d}")
    L.append("")
    L.append("Top LLM-vs-NRC disagreements (LLM said X, NRC said Y):")
    for (le, ne), c in confused.most_common(12):
        L.append(f"  {le:12s} vs {ne:12s} : {c:2d}")
    L.append("")
    L.append("Disagreement examples (LLM | NRC | title):")
    shown = 0
    for r in rows:
        if shown >= 10:
            break
        le, ne = r["llm_emotion"], r["nrc_emotion"]
        if le in EMOTIONS and ne in EMOTIONS and le != ne:
            ttl, txt = texts.get(r["idx"], ("", ""))
            L.append(f"  #{r['idx']+1:3d} {le:12s} | {ne:12s} | {ttl[:55]!r}")
            shown += 1
    report = "\n".join(L)
    print(report)
    (HERE / "emotions_report.txt").write_text(report, encoding="utf-8")
    print("\n[saved] emotions_comparison.csv and emotions_report.txt")

if __name__ == "__main__":
    main()
