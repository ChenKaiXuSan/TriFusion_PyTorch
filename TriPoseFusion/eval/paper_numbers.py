#!/usr/bin/env python3
"""Every number of the T-ITS manuscript's experiment section, computed from the result JSON.

All errors are printed in millimetres.  Sections that depend on runs that do not exist yet are
skipped with a note, so the script can be run while jobs are still in the queue.

    python TriPoseFusion/eval/paper_numbers.py            # everything
    python TriPoseFusion/eval/paper_numbers.py main da    # selected sections
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "logs" / "learned_fusion_experiments"
MM = 1000.0
VIEWS = ("front", "left", "right")
GROUPS = ("head", "shoulders_neck", "body", "hands")
ENVS = {"昼多い": "Day high", "昼少ない": "Day low", "夜多い": "Night high", "夜少ない": "Night low"}
WRISTS = (27, 48)  # right / left wrist in the 52-keypoint order


def folds(tag: str, name: str = "result.json") -> list[dict]:
    out = []
    for fd in sorted((ROOT / tag).glob("fold[0-9]")):
        p = fd / name
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def ms(xs) -> str:
    xs = [MM * x for x in xs]
    return f"{np.mean(xs):6.1f} +- {np.std(xs):4.1f}"


def head(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ------------------------------------------------------------------ in-cabin main table
def sec_main() -> None:
    head("MAIN TABLE (5-fold mean +- std over folds, mm; PCK@100mm; delta = paired per-fold MPJPE / PA change vs mean fusion)")
    ref = folds("eqb_setfusion")
    base_m = np.array([r["training_free"]["fuse_mean"]["mpjpe_m"] for r in ref])
    base_p = np.array([r["training_free"]["fuse_mean"]["pa_mpjpe_m"] for r in ref])

    def line(label, m, p, pck, params=""):
        m, p = np.array(m), np.array(p)
        dm = 100 * (m.mean() / base_m.mean() - 1)
        dp = 100 * (p.mean() / base_p.mean() - 1)
        print(f"{label:34s} {params:>9s}  MPJPE {ms(m)}  PA {ms(p)}  PCK {np.mean(pck):.3f}  dM {dm:+6.1f}%  dPA {dp:+6.1f}%")

    for key in ("single_front", "single_left", "single_right", "best_single_oracle", "fuse_median", "fuse_mean"):
        t = [r["training_free"][key] for r in ref]
        line(key, [x["mpjpe_m"] for x in t], [x["pa_mpjpe_m"] for x in t], [x["pck"]["0.10"] for x in t])
    for tag in ("tf_geomedian", "tf_consensus", "tf_smooth", "tf_gpa", "gating_nt", "gating", "concat_mlp", "skel_tr", "tripose64",
                "tripose128", "lift_mlp", "tcn", "resid64", "setres", "setfusion", "setfusion_d64", "setfusion_d128",
                "setfusion_noaug", "setfusion_nojoint", "setfusion_notemporal"):
        rs = folds(f"eqb_{tag}")
        if len(rs) < 5:
            print(f"{tag:34s} ({len(rs)} folds, skipped)")
            continue
        t = [r["learned_last"] for r in rs]
        line(tag, [x["mpjpe_m"] for x in t], [x["pa_mpjpe_m"] for x in t], [x["pck"]["0.10"] for x in t], f"{rs[0]['params']:,}")


# ------------------------------------------------------------------ joint groups / illumination
def sec_groups() -> None:
    head("JOINT GROUPS (group MPJPE, mm) and PCK at 20/50/100/150 mm")
    ref = folds("eqb_setfusion")
    for key in ("fuse_mean", "best_single_oracle"):
        g = [np.mean([r["training_free"][key]["groups"][x] for r in ref]) for x in GROUPS]
        print(f"{key:22s} " + "  ".join(f"{MM * v:6.1f}" for v in g))
    for tag in ("tcn", "lift_mlp", "resid64", "setfusion"):
        rs = folds(f"eqb_{tag}")
        g = [np.mean([r["learned_last"]["groups"][x] for r in rs]) for x in GROUPS]
        pck = [np.mean([r["learned_last"]["pck"][t] for r in rs]) for t in ("0.02", "0.05", "0.10", "0.15")]
        print(f"{tag:22s} " + "  ".join(f"{MM * v:6.1f}" for v in g) + "   PCK " + " / ".join(f"{v:.3f}" for v in pck))
    mf, sf = np.array([np.mean([r["training_free"]["fuse_mean"]["groups"][x] for r in ref]) for x in GROUPS]), \
        np.array([np.mean([r["learned_last"]["groups"][x] for r in ref]) for x in GROUPS])
    print("setfusion vs mean fusion, per group: " + "  ".join(f"{100 * (a / b - 1):+.1f}%" for a, b in zip(sf, mf)))

    head("BY ILLUMINATION (mean over the 22 sequences of each setting; MPJPE / PA, mm)")
    per = {}
    for tag in ("mean", "setfusion", "tcn", "resid64"):
        rs = folds(f"eqb_{tag}")
        if len(rs) < 5:
            print(f"eqb_{tag}: {len(rs)} folds, skipped")
            continue
        d = {}
        for r in rs:
            for s in r["per_sequence_learned"]:
                d.setdefault(s["env"], []).append((s["metrics"]["mpjpe_m"], s["metrics"]["pa_mpjpe_m"]))
        per[tag] = {e: np.mean(v, axis=0) for e, v in d.items()}
        print(f"{tag:12s} " + "   ".join(f"{ENVS[e]}: {MM * per[tag][e][0]:6.1f} / {MM * per[tag][e][1]:5.1f} (n={len(d[e])})" for e in ENVS)
              + f"   all-sequence mean {MM * np.mean([x[0] for v in d.values() for x in v]):.1f}")
    if "mean" in per and "setfusion" in per:
        print("setfusion vs mean, dMPJPE: " + "  ".join(f"{ENVS[e]} {100 * (per['setfusion'][e][0] / per['mean'][e][0] - 1):+.1f}%" for e in ENVS))


# ------------------------------------------------------------------ robustness
def stress_row(tag: str):
    out = {}
    for mode in ("drop", "zero", "noise"):
        fs = folds(f"eqb_{tag}", f"stress_{mode}.json")
        if len(fs) < 5:
            out[mode] = None
            continue
        err = np.mean([np.mean([f[v]["learned"]["mpjpe_m"] for v in VIEWS]) for f in fs])
        w = np.mean([np.mean([f[v]["weight_on_corrupted_view"] for v in VIEWS]) for f in fs])
        ref = np.mean([np.mean([f[v]["fuse_mean_same_corruption"]["mpjpe_m"] for v in VIEWS]) for f in fs])
        orc = np.mean([np.mean([f[v]["fuse_mean_excluding_view_oracle"]["mpjpe_m"] for v in VIEWS]) for f in fs])
        wrange = (min(np.mean([f[v]["weight_on_corrupted_view"] for f in fs]) for v in VIEWS),
                  max(np.mean([f[v]["weight_on_corrupted_view"] for f in fs]) for v in VIEWS))
        out[mode] = (MM * err, w, MM * ref, MM * orc, wrange)
    return out


def sec_stress() -> None:
    head("ROBUSTNESS (5-fold mean over the three corrupted views; MPJPE mm; w = weight on the corrupted view, per-view range)")
    first = True
    for tag in ("tf_geomedian", "tf_consensus", "gating_nt", "concat_mlp", "skel_tr", "tripose64", "lift_mlp", "tcn", "resid64", "setres",
                "setfusion_noaug", "setfusion"):
        rs = folds(f"eqb_{tag}")
        if len(rs) < 5:
            print(f"{tag:18s} ({len(rs)} folds, skipped)")
            continue
        clean = MM * np.mean([r["learned_last"]["mpjpe_m"] for r in rs])
        st = stress_row(tag)
        if first and st["zero"]:
            ref = {m: st[m] for m in st}
            print(f"{'mean fusion':18s} clean {MM * np.mean([r['training_free']['fuse_mean']['mpjpe_m'] for r in rs]):6.1f}  "
                  + "  ".join(f"{m} {ref[m][2]:6.1f}" if ref[m] else f"{m}   n/a" for m in ("drop", "zero", "noise")))
            print(f"{'oracle exclusion':18s} " + " " * 13
                  + "  ".join(f"{m} {ref[m][3]:6.1f}" if ref[m] else f"{m}   n/a" for m in ("drop", "zero", "noise")))
            first = False
        cells = []
        for m in ("drop", "zero", "noise"):
            if st[m] is None:
                cells.append(f"{m}   n/a")
            else:
                e, w, _, _, wr = st[m]
                cells.append(f"{m} {e:6.1f} ({100 * (e / clean - 1):+5.1f}%, w {w:.3f} [{wr[0]:.2f}-{wr[1]:.2f}])")
        print(f"{tag:18s} clean {clean:6.1f}  " + "  ".join(cells))


# ------------------------------------------------------------------ Drive&Act
def da_cell(rs, key="learned_last"):
    if len(rs) < 3:
        return None
    m, p = [r[key]["mpjpe_m"] for r in rs], [r[key]["pa_mpjpe_m"] for r in rs]
    wr = []
    for r in rs:
        for s in r.get("per_sequence_learned", []):
            pj = s["metrics"].get("per_joint_mpjpe_m") or []
            v = [pj[i] for i in WRISTS if i < len(pj) and pj[i] is not None]
            if v:
                wr.append(np.mean(v))
    return ms(m), ms(p), (MM * np.mean(wr) if wr and key == "learned_last" else float("nan"))


def sec_da() -> None:
    head("DRIVE&ACT, leave-one-subject-out, whole sequences (MPJPE / PA mean +- std over 3 folds, mm; wrists = mean of both wrists)")
    anchor = folds("daeq_setfusion")
    if len(anchor) == 3:
        for key in ("single_front", "single_left", "single_right", "fuse_median", "fuse_mean"):
            c = da_cell(anchor, key=None) if False else None
            m = [r["training_free"][key]["mpjpe_m"] for r in anchor]
            p = [r["training_free"][key]["pa_mpjpe_m"] for r in anchor]
            print(f"{key:14s} MPJPE {ms(m)}  PA {ms(p)}")
    for tag in ("gating_nt", "concat_mlp", "skel_tr", "tripose64", "lift_mlp", "tcn", "resid64", "setres", "setfusion"):
        zs = folds(f"eqb_{tag}", "crosseval_driveandact.json")
        z = (f"zero-shot {MM * np.mean([x['learned_mean']['mpjpe_m'] for x in zs]):5.1f} / {MM * np.mean([x['learned_mean']['pa_mpjpe_m'] for x in zs]):5.1f}"
             f" (PA {MM * min(x['learned_mean']['pa_mpjpe_m'] for x in zs):.1f}-{MM * max(x['learned_mean']['pa_mpjpe_m'] for x in zs):.1f})"
             if zs else "zero-shot n/a")
        cells = []
        for name, pre in (("dataset-ref", "daeq"), ("label-free, official extrinsics", "daeqlf"),
                          ("label-free, self-calibrated extrinsics", "daeqsc"), ("6000", "daeq6k")):
            c = da_cell(folds(f"{pre}_{tag}"))
            cells.append(f"{name}: {c[0]} / {c[1]} wrists {c[2]:.1f}" if c else f"{name}: n/a")
        print(f"{tag:11s} {z} | " + " | ".join(cells))
    for pre in ("daeq", "daeqlf"):
        for tag in ("setfusion", "resid64"):
            rs = folds(f"{pre}_{tag}")
            cal = [r["uncertainty_calibration"] for r in rs if r.get("uncertainty_calibration")]
            if cal:
                print(f"calibration {pre}_{tag}: spearman {np.mean([c['spearman_scale_vs_error'] for c in cal]):.2f}  "
                      f"deciles {MM * np.mean([c['err_most_confident_decile'] for c in cal]):.1f} / {MM * np.mean([c['err_least_confident_decile'] for c in cal]):.1f} mm"
                      f"   view weights f/l/r " + "/".join(f"{np.mean([r['mean_view_weights'][v] for r in rs]):.2f}" for v in VIEWS)
                      + f"   train {np.mean([r['train_seconds'] for r in rs]) / 60:.1f} min")
    for tag in ("setfusion", "resid64", "tcn", "lift_mlp"):
        out = []
        for mode in ("zero", "noise"):
            fs = []
            for fd in sorted((ROOT / f"daeq_{tag}").glob("fold[0-9]")):
                p = fd / f"stress_{mode}.json"
                if p.exists():
                    fs.append(json.loads(p.read_text()))
            if fs:
                out.append(f"{mode} {MM * np.mean([np.mean([f[v]['learned']['mpjpe_m'] for v in VIEWS]) for f in fs]):.1f}"
                           f" (mean fusion {MM * np.mean([np.mean([f[v]['fuse_mean_same_corruption']['mpjpe_m'] for v in VIEWS]) for f in fs]):.1f},"
                           f" oracle {MM * np.mean([np.mean([f[v]['fuse_mean_excluding_view_oracle']['mpjpe_m'] for v in VIEWS]) for f in fs]):.1f})")
        print(f"adapted stress daeq_{tag}: " + "; ".join(out))


# ------------------------------------------------------------------ uncertainty / efficiency
def sec_misc() -> None:
    head("UNCERTAINTY AND TRAINING COST (in-cabin)")
    for tag in ("setfusion", "resid64", "setres", "setfusion_d64", "setfusion_d128", "setfusion_nojoint", "setfusion_notemporal", "setfusion_noaug"):
        rs = folds(f"eqb_{tag}")
        cal = [r["uncertainty_calibration"] for r in rs if r.get("uncertainty_calibration")]
        if len(cal) == 5:
            print(f"{tag:22s} spearman {np.mean([c['spearman_scale_vs_error'] for c in cal]):.3f} "
                  f"[{min(c['spearman_scale_vs_error'] for c in cal):.2f}-{max(c['spearman_scale_vs_error'] for c in cal):.2f}]  "
                  f"deciles {MM * np.mean([c['err_most_confident_decile'] for c in cal]):.1f} / {MM * np.mean([c['err_least_confident_decile'] for c in cal]):.1f}"
                  f" (all {MM * np.mean([c['err_all'] for c in cal]):.1f}) mm   weights f/l/r "
                  + "/".join(f"{np.mean([r['mean_view_weights'][v] for r in rs]):.3f}" for v in VIEWS)
                  + f"   train {np.mean([r['train_seconds'] for r in rs]) / 60:.1f} min/fold")
    for tag in ("gating_nt", "gating"):
        rs = folds(f"eqb_{tag}")
        if rs:
            print(f"{tag:22s} weights f/l/r " + "/".join(f"{np.mean([r['mean_view_weights'][v] for r in rs]):.3f}" for v in VIEWS))


# ------------------------------------------------------------------ paired comparison
def sec_paired() -> None:
    head("PAIRED PER-SEQUENCE DIFFERENCE vs the proposed model (other - proposed, mm; 88 sequences; "
         "subject-level bootstrap, 10000 resamples, 95% interval; negative = other model better)")

    def perseq(tag):
        d = {}
        for r in folds(tag):
            for s in r["per_sequence_learned"]:
                d[(s["person"], s["env"])] = (s["metrics"]["mpjpe_m"], s["metrics"]["pa_mpjpe_m"])
        return d

    base = perseq("eqb_setfusion")
    for tag in ("mean", "resid64", "tcn", "lift_mlp", "setres", "concat_mlp", "skel_tr", "tripose64", "gating_nt",
                "setfusion_nojoint", "setfusion_notemporal", "setfusion_d64", "setfusion_d128"):
        o = perseq(f"eqb_{tag}")
        keys = sorted(set(base) & set(o))
        if len(keys) != 88:
            print(f"{tag:22s} {len(keys)} sequences, skipped")
            continue
        subj = sorted({k[0] for k in keys})
        line = f"{tag:22s}"
        for j, lab in ((0, "MPJPE"), (1, "PA")):
            d = MM * np.array([o[k][j] - base[k][j] for k in keys])
            by = {u: d[[i for i, k in enumerate(keys) if k[0] == u]] for u in subj}
            rng = np.random.default_rng(20261127)
            bs = [np.concatenate([by[u] for u in rng.choice(subj, len(subj))]).mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line += f"  {lab} {d.mean():+6.1f} [{lo:+5.1f}, {hi:+5.1f}] other better on {int((d < 0).sum()):2d}/88"
        print(line)


# ------------------------------------------------------------------ seed variance
SEED_TAGS = ("setfusion", "resid64", "tcn", "lift_mlp")


def seed_runs(tag: str) -> dict[int, str]:
    """seed -> run tag; seed 42 is the original eqb_<tag> run, 43/44 are eqb_<tag>_s43/_s44."""
    return {42: f"eqb_{tag}", 43: f"eqb_{tag}_s43", 44: f"eqb_{tag}_s44"}


def sec_seeds() -> None:
    head("SEED VARIANCE (seeds 42/43/44, common budget; per seed the 5-fold mean, mm)")
    perseq_all = {}
    for tag in SEED_TAGS:
        per_seed, line = {}, f"{tag:12s}"
        for seed, run in seed_runs(tag).items():
            rs = folds(run)
            if len(rs) < 5:
                line += f"  s{seed}: {len(rs)} folds"
                continue
            m = MM * np.mean([r["learned_last"]["mpjpe_m"] for r in rs])
            p = MM * np.mean([r["learned_last"]["pa_mpjpe_m"] for r in rs])
            per_seed[seed] = (m, p)
            line += f"  s{seed} {m:6.1f}/{p:5.1f}"
            for r in rs:
                for s in r["per_sequence_learned"]:
                    perseq_all.setdefault(tag, {}).setdefault((s["person"], s["env"]), {})[seed] = (
                        s["metrics"]["mpjpe_m"], s["metrics"]["pa_mpjpe_m"])
        if len(per_seed) == 3:
            ms_, ps_ = np.array([v[0] for v in per_seed.values()]), np.array([v[1] for v in per_seed.values()])
            line += (f"  | MPJPE {ms_.mean():6.1f} +- {ms_.std(ddof=1):4.1f} ({ms_.min():.1f}-{ms_.max():.1f})"
                     f"  PA {ps_.mean():5.1f} +- {ps_.std(ddof=1):4.1f} ({ps_.min():.1f}-{ps_.max():.1f})")
        print(line)
    head("SEED-AVERAGED PAIRED DIFFERENCE vs setfusion (other - setfusion, mm; per-sequence error averaged over "
         "the 3 seeds; subject-level bootstrap, 10000 resamples, 95% interval; negative = other model better)")
    base = {k: v for k, v in perseq_all.get("setfusion", {}).items() if len(v) == 3}
    for tag in SEED_TAGS[1:]:
        o = {k: v for k, v in perseq_all.get(tag, {}).items() if len(v) == 3}
        keys = sorted(set(base) & set(o))
        if len(keys) != 88:
            print(f"{tag:12s} {len(keys)} sequences with all seeds, skipped")
            continue
        subj = sorted({k[0] for k in keys})
        line = f"{tag:12s}"
        for j, lab in ((0, "MPJPE"), (1, "PA")):
            d = MM * np.array([np.mean([o[k][s][j] for s in (42, 43, 44)]) - np.mean([base[k][s][j] for s in (42, 43, 44)]) for k in keys])
            by = {u: d[[i for i, k in enumerate(keys) if k[0] == u]] for u in subj}
            rng = np.random.default_rng(20261127)
            bs = [np.concatenate([by[u] for u in rng.choice(subj, len(subj))]).mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            same = [MM * np.mean([o[k][s][j] - base[k][s][j] for k in keys]) for s in (42, 43, 44)]
            line += (f"  {lab} {d.mean():+6.1f} [{lo:+5.1f}, {hi:+5.1f}] other better on {int((d < 0).sum()):2d}/88"
                     f" (same-seed pairs {'/'.join(f'{x:+.1f}' for x in same)})")
        print(line)


SECTIONS = {"main": sec_main, "groups": sec_groups, "paired": sec_paired, "stress": sec_stress, "da": sec_da, "misc": sec_misc,
            "seeds": sec_seeds}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[name]()
