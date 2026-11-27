#!/usr/bin/env python3
"""List the millimetre numbers quoted in the prose of main.tex that cannot be traced to a source.

A prose number is considered traced when the same value (to the printed precision) occurs in a
generated table (tables/tab_*.tex) or in the output of TriPoseFusion/eval/paper_numbers.py, passed
on stdin or via --numbers.  Untraced numbers are printed with their context for manual review;
values from Sec. 3 (reference construction) and historical quantities are expected there.

    python TriPoseFusion/eval/paper_numbers.py > /tmp/n.txt && python paper_tits/audit_numbers.py --numbers /tmp/n.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = re.compile(r"(?<![\d.])(\d{1,3}\.\d)(?!\d)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", type=Path, required=True)
    ap.add_argument("--from-section", default=r"\section{Experiments}")
    args = ap.parse_args()
    known = set(NUM.findall(args.numbers.read_text()))
    for t in sorted((HERE / "tables").glob("tab_*.tex")):
        known |= set(NUM.findall(t.read_text()))
    tex = (HERE / "main.tex").read_text()
    body = tex[tex.index(args.from_section):]
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", body, flags=re.S)
    missing = []
    for m in NUM.finditer(body):
        if m.group(1) not in known:
            ctx = body[max(0, m.start() - 55):m.end() + 25].replace("\n", " ")
            missing.append((m.group(1), ctx))
    total = len(NUM.findall(body))
    print(f"{total} one-decimal numbers in the prose after {args.from_section}; {len(missing)} not traced:")
    for v, ctx in missing:
        print(f"  {v:>6s}  ...{ctx}...")


if __name__ == "__main__":
    main()
