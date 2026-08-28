#!/usr/bin/env python3
"""学习型融合基线（rebuttal E 点）：confidence-weighted 逐关节视角加权回归。

与主模型相同的 keypoint 输入（SAM3D 三视角 52 关节，canonical 空间，
与 Table 3 基线协议一致）、相同的 GroupKFold 划分。模型刻意保持极小：

  特征(逐帧·关节·视角): [conf, 与跨视角中位数的距离, 有限性标志]
  MLP(3→32→32→1) → 跨视角 softmax（非有限视角掩掉）→ 加权融合
  可选深度可分离时间卷积残差头（kernel 5）

训练: fold-N train 序列随机 64 帧窗口, masked SmoothL1 vs 伪 GT。
评估: fold-N val 全序列, compute_metrics（与 Table 3 完全同一函数/掩码）。

用法:
  python learned_fusion_baseline.py --cache-dir ... --fold 0 \
      --output-dir TriPoseFusion/eval/logs/learned_fusion_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_fusion_baselines_pesudo_gt import compute_metrics, fuse_views  # noqa: E402

ENV_KEYS = {"夜多い", "夜少ない", "昼多い", "昼少ない"}


# ---------------------------------------------------------------- data


def load_fold_pairs(index_mapping: Path, fold: int) -> tuple[list, list]:
    with open(index_mapping / f"fold_{fold}.json", "r", encoding="utf-8") as f:
        d = json.load(f)

    def pairs(split):
        return sorted({(s["person_id"], s["env_folder"]) for s in d[split]})

    return pairs("train"), pairs("val")


class SeqStore:
    """把缓存 npz 载入内存（fp16 -> fp32 视需转换）。"""

    def __init__(self, cache_dir: Path, pairs: list[tuple[str, str]]) -> None:
        self.seqs = []
        missing = []
        for person, env in pairs:
            p = cache_dir / f"{person}_{env}.npz"
            if not p.exists():
                missing.append(p.name)
                continue
            with np.load(p, allow_pickle=False) as z:
                self.seqs.append(
                    {
                        "person": person,
                        "env": env,
                        "view_pose": z["view_pose"].astype(np.float32),  # (T,J,V,3)
                        "view_conf": z["view_conf"].astype(np.float32),  # (T,J,V)
                        "gt_pose": z["gt_pose"].astype(np.float32),  # (T,J,3)
                        "gt_valid": z["gt_valid"].astype(bool),  # (T,J)
                    }
                )
        if missing:
            print(f"WARNING: {len(missing)} cache files missing: {missing[:4]} ...")
        if not self.seqs:
            raise SystemExit("no cached sequences loaded")

    def sample_windows(
        self, rng: np.random.Generator, batch: int, window: int
    ) -> dict[str, torch.Tensor]:
        vp, vc, gp, gv = [], [], [], []
        for _ in range(batch):
            s = self.seqs[rng.integers(len(self.seqs))]
            t_total = s["view_pose"].shape[0]
            t0 = int(rng.integers(0, max(t_total - window, 1)))
            sl = slice(t0, t0 + window)
            vp.append(s["view_pose"][sl])
            vc.append(s["view_conf"][sl])
            gp.append(s["gt_pose"][sl])
            gv.append(s["gt_valid"][sl])
        return {
            "view_pose": torch.from_numpy(np.stack(vp)),
            "view_conf": torch.from_numpy(np.stack(vc)),
            "gt_pose": torch.from_numpy(np.stack(gp)),
            "gt_valid": torch.from_numpy(np.stack(gv)),
        }


# ---------------------------------------------------------------- model


class LearnedFusion(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        temporal: bool = True,
        kernel: int = 5,
        num_joints: int = 52,
        joint_emb: int = 8,
    ) -> None:
        super().__init__()
        self.joint_emb = nn.Embedding(num_joints, joint_emb) if joint_emb > 0 else None
        in_dim = 3 + (joint_emb if joint_emb > 0 else 0)
        self.weight_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.temporal = temporal
        if temporal:
            self.tconv = nn.Conv1d(3, 3, kernel_size=kernel, padding=kernel // 2, groups=3)
            nn.init.zeros_(self.tconv.weight)
            nn.init.zeros_(self.tconv.bias)

    def forward(
        self, view_pose: torch.Tensor, view_conf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # view_pose (B,T,J,V,3), view_conf (B,T,J,V)
        finite = torch.isfinite(view_pose).all(dim=-1)  # (B,T,J,V)
        pose = torch.nan_to_num(view_pose, nan=0.0)
        n_finite = finite.sum(dim=-1, keepdim=True).clamp_min(1)  # (B,T,J,1)
        mean_pose = (pose * finite.unsqueeze(-1)).sum(dim=-2) / n_finite  # (B,T,J,3)
        disagreement = torch.linalg.norm(
            pose - mean_pose.unsqueeze(-2), dim=-1
        ) * finite  # (B,T,J,V)
        conf = torch.nan_to_num(view_conf, nan=0.0)
        feats = torch.stack([conf, disagreement, finite.float()], dim=-1)
        if self.joint_emb is not None:
            b, t, j, v, _ = feats.shape
            emb = self.joint_emb(
                torch.arange(j, device=feats.device)
            )[None, None, :, None, :].expand(b, t, j, v, -1)
            feats = torch.cat([feats, emb], dim=-1)
        logits = self.weight_mlp(feats).squeeze(-1)  # (B,T,J,V)
        logits = logits.masked_fill(~finite, -1e4)
        weights = torch.softmax(logits, dim=-1)
        fused = (pose * weights.unsqueeze(-1)).sum(dim=-2)  # (B,T,J,3)
        if self.temporal:
            b, t, j, c = fused.shape
            x = fused.permute(0, 2, 3, 1).reshape(b * j, c, t)
            fused = fused + self.tconv(x).reshape(b, j, c, t).permute(0, 3, 1, 2)
        # 无任何有限视角的关节输出 NaN，评估侧照常剔除
        any_finite = finite.any(dim=-1, keepdim=True)
        fused = torch.where(any_finite.expand_as(fused) > 0, fused, torch.nan)
        return fused, weights


# ---------------------------------------------------------------- train/eval


def masked_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid & torch.isfinite(gt).all(dim=-1) & torch.isfinite(pred).all(dim=-1)
    if not mask.any():
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[mask], gt[mask], beta=0.05)


def evaluate(model: LearnedFusion, store: SeqStore, pck_thresholds) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for s in store.seqs:
            vp = torch.from_numpy(s["view_pose"]).unsqueeze(0)
            vc = torch.from_numpy(s["view_conf"]).unsqueeze(0)
            pred, weights = model(vp, vc)
            metrics = compute_metrics(
                pred=pred.squeeze(0).numpy(),
                gt=s["gt_pose"],
                valid_mask=s["gt_valid"],
                pck_thresholds=tuple(pck_thresholds),
            )
            if metrics:
                rows.append(
                    {
                        "person": s["person"],
                        "env": s["env"],
                        "metrics": metrics,
                        "mean_view_weights": weights.squeeze(0).mean(dim=(0, 1)).tolist(),
                    }
                )
    model.train()
    return {"rows": rows}


def fixed_fusion_anchor(store: SeqStore, pck_thresholds) -> list[dict]:
    """同一缓存数据上的固定 confidence 加权融合（Table 3 同款），作对照锚点。"""
    rows = []
    for s in store.seqs:
        pred = fuse_views(s["view_pose"], s["view_conf"], "confidence")
        m = compute_metrics(
            pred=pred,
            gt=s["gt_pose"],
            valid_mask=s["gt_valid"],
            pck_thresholds=tuple(pck_thresholds),
        )
        if m:
            rows.append({"person": s["person"], "env": s["env"], "metrics": m})
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = ("mpjpe_m", "pa_mpjpe_m", "mpjpe_pa_frames_m", "root_mpjpe_m")
    agg = {}
    for k in keys:
        vals = [r["metrics"].get(k) for r in rows if r["metrics"].get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    pck_all = {}
    for th in ("0.02", "0.05", "0.1", "0.15"):
        vals = [
            r["metrics"]["pck"].get(th)
            for r in rows
            if isinstance(r["metrics"].get("pck"), dict) and r["metrics"]["pck"].get(th) is not None
        ]
        if vals:
            pck_all[th] = float(np.mean(vals))
    agg["pck"] = pck_all
    agg["num_sequences"] = len(rows)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/work/1/SKIING/chenkaixu/data/drive/learned_fusion_cache"),
    )
    parser.add_argument(
        "--index-mapping",
        type=Path,
        default=Path("/work/1/SKIING/chenkaixu/data/drive/index_mapping"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--no-temporal", dest="temporal", action="store_false")
    parser.set_defaults(temporal=True)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--pck-thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "logs" / "learned_fusion_baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_pairs, val_pairs = load_fold_pairs(args.index_mapping, args.fold)
    print(f"fold {args.fold}: {len(train_pairs)} train / {len(val_pairs)} val seqs")
    train_store = SeqStore(args.cache_dir, train_pairs)
    val_store = SeqStore(args.cache_dir, val_pairs)

    anchor_rows = fixed_fusion_anchor(val_store, args.pck_thresholds)
    anchor_agg = summarize(anchor_rows)
    print(f"fixed-confidence anchor (val): {json.dumps(anchor_agg)}")

    model = LearnedFusion(hidden=args.hidden, temporal=args.temporal)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = None
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = train_store.sample_windows(rng, args.batch, args.window)
        pred, _ = model(batch["view_pose"], batch["view_conf"])
        loss = masked_loss(pred, batch["gt_pose"], batch["gt_valid"])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, val_store, args.pck_thresholds)
            agg = summarize(ev["rows"])
            history.append({"step": step, "loss": float(loss.item()), **agg})
            print(
                f"step {step} loss {loss.item():.4f} "
                f"val MPJPE {agg['mpjpe_m']:.4f} PA {agg['pa_mpjpe_m']:.4f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
            if best is None or agg["mpjpe_m"] < best["agg"]["mpjpe_m"]:
                best = {"step": step, "agg": agg, "rows": ev["rows"],
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], args.output_dir / f"learned_fusion_fold{args.fold}.pt")
    payload = {
        "fold": args.fold,
        "params": n_params,
        "config": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "best_step": best["step"],
        "best_val": best["agg"],
        "fixed_confidence_anchor_val": anchor_agg,
        "history": history,
        "per_sequence": best["rows"],
    }
    with open(args.output_dir / f"learned_fusion_fold{args.fold}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(best["agg"], indent=1))
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
