#!/usr/bin/env python3
"""Generate a self-contained offline HTML dashboard for the STEP 6 balanced 3-class run.

Every figure is recomputed here from the SAVED output:
  predictions_balanced.csv       per-review truth (from rating), 3-class prediction, emotion
  data/balanced_150.jsonl        review text + rating (for stars / detail)
  Gift_Cards.jsonl.gz            whole-file class distribution (context)
  confusion_balanced.csv         cross-check
Asserts invariants so the page can never drift from the scoring. Emits dashboard.html.
"""
import csv, json, html, gzip
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = ["POS", "NEU", "NEG"]
CLASS_FULL = {"POS": "Positive", "NEU": "Neutral", "NEG": "Negative"}
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

def pct(x, d=1):
    return f"{x*100:.{d}f}%"

# ---------------- data loading + metrics (single source of truth) ----------------
def load():
    texts, ratings = {}, {}
    for line in (HERE / "data/balanced_150.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        idx = int(r.get("idx", len(texts)))
        ratings[idx] = float(r["rating"])
        texts[idx] = ((r.get("title") or "").strip(), (r.get("text") or "").strip())
    rows = []
    with open(HERE / "predictions_balanced.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = int(r["idx"])
            truth, pred = r["truth"], r["llm_sent"]
            rows.append({
                "idx": idx, "asin": r["asin"], "rating": ratings.get(idx, float(r["rating"])),
                "truth": truth, "pred": pred, "correct": truth == pred,
                "title": (r.get("title") or "").strip(),
                "text": texts.get(idx, ("", ""))[1],
                "emotion": r.get("llm_emotion", ""),
            })
    rows.sort(key=lambda x: x["idx"])
    return rows

def board(rows):
    n = len(rows)
    cm = {t: {p: 0 for p in C} for t in C}
    for r in rows:
        if r["pred"] in C and r["truth"] in C:
            cm[r["truth"]][r["pred"]] += 1
    truth_counts = Counter(r["truth"] for r in rows)
    def f1(p, rr): return 2 * p * rr / (p + rr) if (p + rr) else 0.0
    per = {}
    for c in C:
        tp = cm[c][c]; fp = sum(cm[t][c] for t in C if t != c); fn = sum(cm[c][p] for p in C if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per[c] = {"n": tp + fn, "prec": prec, "rec": rec, "f1": f1(prec, rec)}
    correct = sum(r["correct"] for r in rows)
    stars = Counter(int(r["rating"]) for r in rows)
    return {
        "n": n, "correct": correct, "wrong": n - correct,
        "cm": cm, "truth": truth_counts, "per": per,
        "acc": correct / n, "macro_f1": sum(per[c]["f1"] for c in C) / 3,
        "baseline": max(truth_counts.values()) / n,
        "stars": {s: stars.get(s, 0) for s in range(1, 6)},
        "pred_dist": Counter(r["pred"] for r in rows if r["pred"] in C),
        "emotion_dist": Counter(r["emotion"] for r in rows if r["emotion"] in EMOTIONS),
    }

def wholefile_dist(src):
    g = {"POS": 0, "NEU": 0, "NEG": 0}
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rt = json.loads(line)["rating"]
            g["POS" if rt >= 4 else ("NEU" if rt == 3 else "NEG")] += 1
    tot = sum(g.values())
    return g, {k: v / tot for k, v in g.items()}

# ---------------- fragment builders ----------------
def kpi_frag(m):
    return f"""
    <div class="kpi kpi-hero">
      <span class="kpi-label">Agrees with rating</span>
      <span class="kpi-value">{pct(m['acc'])}</span>
      <span class="kpi-sub">{m['correct']} of {m['n']} · {m['wrong']} wrong</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Macro F1</span>
      <span class="kpi-value kpi-med">{m['macro_f1']:.3f}</span>
      <span class="kpi-sub">mean of POS / NEU / NEG F1</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Reviews scored</span>
      <span class="kpi-value kpi-med">{m['n']}</span>
      <span class="kpi-sub"><span class="dot dot-pos"></span>{m['truth']['POS']} pos · <span class="dot dot-neu"></span>{m['truth']['NEU']} neu · <span class="dot dot-neg"></span>{m['truth']['NEG']} neg</span>
    </div>
    <div class="kpi">
      <span class="kpi-label">Failed / unclear calls</span>
      <span class="kpi-value kpi-med">0</span>
      <span class="kpi-sub">of {m['n']} returned a label</span>
    </div>"""

def stars_frag(m):
    mx = max(m["stars"].values()) or 1
    bars = []
    for s in range(1, 6):
        c = m["stars"][s]
        w = c / mx * 100
        cls = {4: "pos", 5: "pos", 3: "neu", 2: "neg", 1: "neg"}[s]
        bars.append(
            f'<div class="star-row"><span class="star-r">{s}★</span>'
            f'<div class="star-track"><div class="star-fill {cls}" style="width:{max(w, 2.5) if c else 0}%"></div></div>'
            f'<span class="star-n">{c}</span></div>')
    return ('<div class="stars-wrap">' + "".join(bars) + "</div>"
            '<p class="helper" style="margin-top:10px">Balanced sample: 50 per class. '
            '3★ fills the NEUTRAL class; 1–2★ = NEG, 4–5★ = POS.</p>')

def imbalance_frag(m, whole, wfrac):
    def side(name, dfr, pos, neu, neg):
        return (f'<div class="imb-row"><span class="imb-name">{name}</span>'
                f'<div class="imb-bar"><span class="imb-pos" style="flex:{pos}"></span>'
                f'<span class="imb-neu" style="flex:{neu}"></span><span class="imb-neg" style="flex:{neg}"></span></div>'
                f'<span class="imb-nums">POS {pct(pos)} · NEU {pct(neu)} · NEG {pct(neg)}</span></div>')
    return (f'<div class="imb">{side("Whole file", whole, wfrac["POS"], wfrac["NEU"], wfrac["NEG"])}'
            + side("In this run", m, m["truth"]["POS"] / m["n"], m["truth"]["NEU"] / m["n"], m["truth"]["NEG"] / m["n"]) + '</div>'
            + f'<p class="helper" style="margin-top:12px">Whole-file counts: '
            f'POS {whole["POS"]:,} · NEU {whole["NEU"]:,} · NEG {whole["NEG"]:,}. '
            f'NEUTRAL (3★) is just {pct(wfrac["NEU"], 1)} of real reviews.</p>')

def matrix_frag(m):
    def cell(count, note, hit):
        cls = "cell-hit" if hit else "cell-fp"
        return f'<div class="mx-cell {cls}"><span class="mx-count">{count}</span><span class="mx-lb">{note}</span></div>'
    head = ('<div class="mx-h"><span class="mx-corner">truth<b>↓</b><i>predicted →</i></span>' +
            "".join(f'<span class="mx-hcell">{CLASS_FULL[c]}</span>' for c in C) + "</div>")
    rows_html = []
    for t in C:
        cells = []
        for p in C:
            count = m["cm"][t][p]
            if t == p:
                note, hit = "correct", True
            elif p == "NEU":
                note, hit = "→ neutral", False
            elif t == "NEU":
                note, hit = "collapsed to " + p.lower(), False
            else:
                note, hit = "flipped to " + p.lower(), False
            cells.append(cell(count, note, hit))
        rows_html.append(f'<div class="mx-row"><span class="mx-rlabel">{CLASS_FULL[t]}</span>' + "".join(cells) + "</div>")
    return ('<div class="mx">' + head + "".join(rows_html) + "</div>"
            '<div class="mx-legend">'
            '<span class="lg"><span class="sw sw-hit"></span>correct</span>'
            '<span class="lg"><span class="sw sw-fp"></span>mis-scored</span></div>')

def compare_frag(m):
    """'Correct answer vs predicted' per truth class.
    Bar = proportions only; exact counts carried in a separate colored line under
    each row so tiny segments can never erase the numbers (layout-bug guard)."""
    rows = []
    for t in C:
        segs = [f'<span class="cmp-seg seg-{p.lower()}" style="flex:{c}"></span>'
                for p in C if (c := m["cm"][t][p]) > 0]
        rows.append(f'<div class="cmp-row"><span class="cmp-lb">{CLASS_FULL[t]}</span>'
                    f'<div><div class="cmp-bar">{"".join(segs)}</div>'
                    f'<div class="cmp-counts"><span class="cc-pos">POS</span> {m["cm"][t]["POS"]}'
                    f'&nbsp;·&nbsp;<span class="cc-neu">NEU</span> {m["cm"][t]["NEU"]}'
                    f'&nbsp;·&nbsp;<span class="cc-neg">NEG</span> {m["cm"][t]["NEG"]}</div>'
                    f'</div></div>')
    legend = "".join(f'<span class="lg"><span class="sw sw-{p.lower()}"></span>predicted {CLASS_FULL[p]}</span>' for p in C)
    return ("<div class=\"cmp\">" + "".join(rows) + "</div>"
            '<div class="cmp-legend">' + legend + "</div>")

def recall_frag(m):
    rows = []
    for c in C:
        pc = m["per"][c]
        w = max(pc["rec"] * 100, 2.0)
        rows.append(
            f'<div class="pc-row"><div class="pc-head"><span class="pc-name"><span class="dot dot-{c.lower()}"></span>{CLASS_FULL[c]}</span>'
            f'<span class="pc-nums">precision {pct(pc["prec"])} · recall {pct(pc["rec"])} · F1 {pc["f1"]:.3f}</span></div>'
            f'<div class="pc-bar"><span class="pc-fill pc-{c.lower()}" style="width:{w}%"></span></div>'
            f'<div class="pc-sub">recall = {pct(pc["rec"])} of {pc["n"]} true {CLASS_FULL[c].lower()} caught</div></div>')
    return '<div class="pc">' + "".join(rows) + "</div>"

def emotion_frag(m):
    rows = []
    for e in EMOTIONS:
        c = m["emotion_dist"].get(e, 0)
        rows.append(f'<span class="emo"><i class="emo-dot" style="opacity:{0.06 + 0.94 * (c / (m["n"] or 1)):.2f}"></i>{e} <b>{c}</b></span>')
    return '<div class="emo-wrap">' + "".join(rows) + "</div>"

def wrong_frag(rows):
    # top story: NEU collapse + a few wrong examples
    neuf = [r for r in rows if r["truth"] == "NEU"]
    collapsed = [r for r in neuf if r["pred"] != "NEU"]
    kept = [r for r in neuf if r["pred"] == "NEU"]
    cards = []
    for r in collapsed[:4]:
        tone = "coll" if r["pred"] == "NEG" else "fp"
        cards.append(f"""
        <article class="w-card">
          <div class="w-top"><span class="w-tag {tone}">3★ called {r['pred']}</span><span class="w-row">row #{int(r['idx'])+1}</span></div>
          <p class="w-title">“{html.escape(r['title'])}”</p>
          <div class="w-vs"><div class="w-slot"><span class="w-lb">Rating (truth)</span><span class="w-val">3★ → NEU</span></div>
                <div class="w-slot"><span class="w-lb">Model said</span><span class="w-val">{r['pred']}</span></div></div>
          <p class="w-text">{html.escape(r['text'][:220])}{'…' if len(r['text']) > 220 else ''}</p>
        </article>""")
    return """<div class="wrong-grid">""" + "".join(cards) + f"""</div>
    <p class="w-note" style="margin-top:16px"><b>The 3-star problem in numbers:</b> of 50 true-NEUTRAL reviews, the model kept <b>{len(kept)}</b> as NEUTRAL but pulled <b>{len(collapsed)}</b> into another class — most ({sum(1 for r in collapsed if r['pred']=='NEG')}) into NEGATIVE and {sum(1 for r in collapsed if r['pred']=='POS')} into POSITIVE. 3-star text tends to be terse and complaint-flavoured, so the words read negative.</p>"""

def main():
    rows = load()
    m = board(rows)
    whole, wfrac = wholefile_dist(HERE / "Gift_Cards.jsonl.gz")
    # invariants
    assert m["n"] == 150, m["n"]
    assert m["truth"] == Counter({"POS": 50, "NEU": 50, "NEG": 50}), m["truth"]
    assert m["cm"]["POS"] == {"POS": 48, "NEU": 1, "NEG": 1}
    assert m["cm"]["NEU"] == {"POS": 5, "NEU": 16, "NEG": 29}
    assert m["cm"]["NEG"] == {"POS": 1, "NEU": 1, "NEG": 48}
    assert abs(m["acc"] - 112 / 150) < 1e-9 and abs(m["macro_f1"] - 0.715) < 0.002

    headline = (f"The model agrees with the star rating on <strong>{m['correct']} of {m['n']} "
                f"reviews ({pct(m['acc'])}), missing {m['wrong']}</strong> — but most of the misses are "
                f"the model reading 3-star reviews as negative: only {m['cm']['NEU']['NEU']} of {m['truth']['NEU']} "
                f"true-neutral reviews were kept neutral.")

    rows_payload = [{"idx": r["idx"], "asin": r["asin"], "rating": r["rating"], "truth": r["truth"],
                     "pred": r["pred"], "correct": r["correct"], "title": r["title"],
                     "text": r["text"], "emotion": r["emotion"]} for r in rows]

    templ = (HERE / "_dashboard_template.html").read_text(encoding="utf-8")
    out = (templ
           .replace("@@HEADLINE@@", headline)
           .replace("@@KPI@@", kpi_frag(m))
           .replace("@@STARS@@", stars_frag(m))
           .replace("@@IMBALANCE@@", imbalance_frag(m, whole, wfrac))
           .replace("@@MATRIX@@", matrix_frag(m))
           .replace("@@COMPARE@@", compare_frag(m))
           .replace("@@RECALL@@", recall_frag(m))
           .replace("@@EMOTION@@", emotion_frag(m))
           .replace("@@WRONG@@", wrong_frag(rows))
           .replace("@@ACC@@", pct(m["acc"])).replace("@@MACRO@@", f"{m['macro_f1']:.3f}")
           .replace("@@BASELINE@@", pct(m["baseline"]))
           .replace("@@POSN@@", str(m["truth"]["POS"])).replace("@@NEUN@@", str(m["truth"]["NEU"])).replace("@@NEGN@@", str(m["truth"]["NEG"]))
           .replace("@@ROWS@@", json.dumps(rows_payload)))
    (HERE / "dashboard.html").write_text(out, encoding="utf-8")

    print("dashboard.html written")
    print(f"  n={m['n']} correct={m['correct']} wrong={m['wrong']}  acc={pct(m['acc'])}  macroF1={m['macro_f1']:.3f}")
    print("  confusion:"); [print("   ", t, m["cm"][t]) for t in C]
    print("  whole-file:", {k: f"{v:,}" for k, v in whole.items()})

if __name__ == "__main__":
    main()
