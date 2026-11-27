#!/usr/bin/env python3
"""Archive superseded fused-keypoint fields out of the downstream npz files.

Two phases, run separately so the destructive step only happens after verification:

  extract   copy the fields of one export (fused_<sfx>, uncertainty_<sfx>, gate_<sfx>, fold_<sfx>)
            together with frame_ids / cameras into slim files <archive-dir>/<sequence>.npz and
            verify every archived array bit-for-bit against the source.  Read-only on the source.
  strip     for every sequence whose archive copies verify, rewrite the source file without the
            archived fields (atomic: temporary file + rename) and check that every remaining
            field is unchanged.  Refuses a sequence if any requested archive is missing or differs.

    python TriPoseFusion/eval/archive_fused_fields.py extract --suffix v3   --archive-dir <dir>
    python TriPoseFusion/eval/archive_fused_fields.py strip   --suffix v3 full --archive-dirs <dir_v3> <dir_full>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SRC = Path("/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/fused_keypoints")
PARTS = ("fused", "uncertainty", "gate", "fold")
KEEP_WITH_ARCHIVE = ("frame_ids", "cameras")


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)


def fields(sfx: str, present) -> list[str]:
    return [f"{p}_{sfx}" for p in PARTS if f"{p}_{sfx}" in present]


def verify_archive(src_npz, arc_path: Path, sfx: str) -> bool:
    if not arc_path.exists():
        return False
    with np.load(arc_path, allow_pickle=True) as arc:
        names = fields(sfx, src_npz.files) + [k for k in KEEP_WITH_ARCHIVE if k in src_npz.files]
        return all(k in arc.files and same(src_npz[k], arc[k]) for k in names)


def cmd_extract(args) -> None:
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(args.src.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p, allow_pickle=True) as z:
            names = fields(args.suffix, z.files)
            if not names:
                print(f"  {p.stem}: no *_{args.suffix} fields, skipped")
                continue
            out = args.archive_dir / p.name
            if not (out.exists() and verify_archive(z, out, args.suffix)):
                payload = {k: z[k] for k in names + [k for k in KEEP_WITH_ARCHIVE if k in z.files]}
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, **payload)
                os.replace(tmp, out)
            if not verify_archive(z, out, args.suffix):
                raise SystemExit(f"!! verification failed for {out}")
            n += 1
    print(f"extract {args.suffix}: {n} sequences archived and verified bit-identical in {args.archive_dir}")


def cmd_strip(args) -> None:
    if len(args.suffix) != len(args.archive_dirs):
        raise SystemExit("one --archive-dirs entry per --suffix")
    done = skipped = 0
    for p in sorted(args.src.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p, allow_pickle=True) as z:
            drop = [k for sfx in args.suffix for k in fields(sfx, z.files)]
            if not drop:
                skipped += 1
                continue
            for sfx, adir in zip(args.suffix, args.archive_dirs):
                if fields(sfx, z.files) and not verify_archive(z, adir / p.name, sfx):
                    raise SystemExit(f"!! refusing to strip {p.name}: archive of *_{sfx} missing or different")
            if args.require and any(k not in z.files for k in args.require):
                raise SystemExit(f"!! refusing to strip {p.name}: required field missing ({args.require})")
            keep = {k: z[k] for k in z.files if k not in drop}
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **keep)
        with np.load(tmp, allow_pickle=True) as t:
            if sorted(t.files) != sorted(keep) or not all(same(keep[k], t[k]) for k in keep):
                os.remove(tmp)
                raise SystemExit(f"!! rewritten file differs for {p.name}; original left untouched")
        os.replace(tmp, p)
        done += 1
    print(f"strip {args.suffix}: {done} files rewritten without the archived fields, {skipped} already clean")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--suffix", required=True)
    e.add_argument("--archive-dir", type=Path, required=True)
    e.set_defaults(func=cmd_extract)
    s = sub.add_parser("strip")
    s.add_argument("--suffix", nargs="+", required=True)
    s.add_argument("--archive-dirs", type=Path, nargs="+", required=True)
    s.add_argument("--require", nargs="*", default=["fused_main"], help="fields that must be present before stripping")
    s.set_defaults(func=cmd_strip)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
