#!/usr/bin/env python3
"""Extract the first N rows from the Amazon Gift Cards gzipped JSONL.
Saves them (rating included, for after-the-fact checking only) to a plain jsonl.
"""
import gzip, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="path to .jsonl.gz")
    ap.add_argument("dst", help="output .jsonl")
    ap.add_argument("-n", "--nrows", type=int, default=100)
    args = ap.parse_args()

    count = 0
    with gzip.open(args.src, "rt", encoding="utf-8") as fin, \
         open(args.dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(line + "\n")
            count += 1
            if count >= args.nrows:
                break

    # sanity: report per-class counts using the ground-truth rule (rating>=4 positive)
    import collections
    dist = collections.Counter()
    with open(args.dst, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dist["pos" if d["rating"] >= 4 else "neg"] += 1
    print(f"wrote {count} rows to {args.dst}")
    print("ground-truth class distribution (rating>=4 -> positive):",
          dict(sorted(dist.items())))

if __name__ == "__main__":
    main()
