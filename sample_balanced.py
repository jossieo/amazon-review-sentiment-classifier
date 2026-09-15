#!/usr/bin/env python3
"""Step 6 — three-class balanced sample from the whole Gift Cards file.

- Full-file rating/class distribution.
- A fixed-seed (42) random sample of ~50 per class from the WHOLE file so the
  rarer classes (NEUTRAL, NEGATIVE) are not under-represented. Same set every run.
- Writes data/balanced_150.jsonl and a meta file recording the sampled asins+seed.
"""
import gzip, json, random, argparse

CLASS_OF = lambda r: "POS" if r >= 4 else ("NEU" if r == 3 else "NEG")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", default="Gift_Cards.jsonl.gz", nargs="?")
    ap.add_argument("--out", default="data/balanced_150.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=50)
    args = ap.parse_args()

    groups = {"POS": [], "NEU": [], "NEG": []}
    dist = {}
    idx = 0
    with gzip.open(args.src, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cls = CLASS_OF(r["rating"])
            dist[cls] = dist.get(cls, 0) + 1
            groups[cls].append((idx, r))
            idx += 1

    print("full-file review counts by class (rating 4-5 POS / 3 NEU / 1-2 NEG):")
    total = sum(dist.values())
    for c in ["POS", "NEU", "NEG"]:
        n = dist.get(c, 0)
        print(f"  {c:4s} {n:7d}  ({n/total*100:5.1f}%)")

    rng = random.Random(args.seed)
    chosen = []
    meta = {"seed": args.seed, "per_class_target": args.per_class, "classes": {}}
    for c in ["POS", "NEU", "NEG"]:
        pool = groups[c]
        k = min(args.per_class, len(pool))
        picked = rng.sample(pool, k)
        chosen.append((c, picked))
        meta["classes"][c] = {"available": len(pool), "sampled": len(picked)}

    with open(args.out, "w", encoding="utf-8") as f:
        for c, picked in chosen:
            for idx2, rec in picked:
                f.write(json.dumps(rec) + "\n")

    # companion meta file
    meta["sample_asins"] = {c: [p[1]["asin"] for p in picked]
                            for c, picked in chosen}
    meta_path = args.out.replace(".jsonl", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved balanced sample to {args.out} ({sum(len(p) for _, p in chosen)} reviews)")
    for c, picked in chosen:
        print(f"  {c:4s} sampled {len(picked)}")
    print(f"meta (seed, asins) -> {meta_path}")
    # quick re-read sanity: class balance of the file we wrote
    from collections import Counter
    with open(args.out, encoding="utf-8") as f:
        bal = Counter(CLASS_OF(json.loads(l)["rating"]) for l in f if l.strip())
    print("sanity check — classes in written file:", dict(bal))

if __name__ == "__main__":
    main()
