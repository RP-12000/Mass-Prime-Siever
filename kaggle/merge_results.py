#!/usr/bin/env python3
"""Combine results.json files from several sieve runs of the same sequence and n range.

Each run only removes the n that have a factor in its own prime range, so the candidates that
are left after all runs are the n listed as "prp" in every file (the intersection).

    python merge_results.py gpu0/results.json gpu1/results.json [laptop/results.json ...] -o candidates.txt
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", default="candidates.txt")
    args = ap.parse_args()

    survivors = None
    key = None
    ranges = []
    for path in args.files:
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            this_key = (int(entry["b"]), int(entry["c"]), int(entry["N0"]), int(entry["N1"]))
            if key is None:
                key = this_key
            elif this_key != key:
                sys.exit(f"{path}: b, c, N0, N1 = {this_key} differ from {key}; not merging")
            prp = {int(n) for n in entry["prp"]}
            survivors = prp if survivors is None else survivors & prp
            ranges.append((int(entry["P0"]), int(entry["P1"]), path))

    ranges.sort()
    print("Prime ranges covered:")
    for p0, p1, path in ranges:
        print(f"  [{p0}, {p1})  {path}")
    for (a0, a1, _), (b0, b1, _) in zip(ranges, ranges[1:]):
        if b0 > a1:
            print(f"  WARNING: gap [{a1}, {b0}) was not sieved")
    with open(args.output, "w") as f:
        for n in sorted(survivors):
            f.write(f"{n}\n")
    b, c, n0, n1 = key
    print(f"{len(survivors)} candidates left for {b}^n{c:+d}, n in [{n0}, {n1}) -> {args.output}")


if __name__ == "__main__":
    main()
