"""Regression tests for the reviewer-reported metric and canonicalization bugs.

Covers three fixes:
1. gate_entropy_loss: the old code clamped log(alpha) instead of alpha, making
   the regularizer a zero-gradient constant log(V).
2. procrustes_align (numpy + torch): the old code applied the transposed
   rotation and a norm-ratio scale, systematically over-estimating PA-MPJPE
   (it could even exceed MPJPE, as reviewers observed in Table 3).
3. Canonicalization shoulder fallback: replacing both shoulders with the neck
   produced a zero x-axis, a rank-zero rotation, and an all-zero canonical pose.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TPF_ROOT = REPO_ROOT / "TriPoseFusion"
for path in (str(REPO_ROOT), str(TPF_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from TriPoseFusion.eval.eval_fusion_baselines_pesudo_gt import (  # noqa: E402
    canonicalize_pose,
    compute_metrics,
    procrustes_align,
)
from models.keypoint_mlp import RobustCanonicalization  # noqa: E402
from trainer.train_triple_fusion import gate_entropy_loss  # noqa: E402


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(0.2, 2.8))
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class GateEntropyLossTests(unittest.TestCase):
    def test_uniform_alpha_gives_zero_loss(self) -> None:
        alpha = torch.full((2, 4, 5, 3), 1.0 / 3.0)
        self.assertAlmostEqual(float(gate_entropy_loss(alpha)), 0.0, places=5)

    def test_one_hot_alpha_gives_log_v(self) -> None:
        alpha = torch.zeros(1, 1, 1, 3)
        alpha[..., 0] = 1.0
        self.assertAlmostEqual(
            float(gate_entropy_loss(alpha)), float(np.log(3.0)), places=3
        )

    def test_loss_has_nonzero_gradient(self) -> None:
        logits = torch.tensor([[0.5, 0.2, -0.3]], requires_grad=True)
        loss = gate_entropy_loss(torch.softmax(logits, dim=-1))
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 1e-6)


class ProcrustesAlignTests(unittest.TestCase):
    def test_recovers_similarity_transform(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(5):
            gt = rng.normal(size=(20, 3))
            rot = _random_rotation(rng)
            noise = 0.01 * rng.normal(size=gt.shape)
            src = 1.7 * (gt @ rot.T) + rng.normal(size=3) + noise
            aligned = procrustes_align(src, gt)
            pa_err = float(np.linalg.norm(aligned - gt, axis=-1).mean())
            raw_err = float(np.linalg.norm(src - gt, axis=-1).mean())
            self.assertLess(pa_err, 0.1)
            self.assertLess(pa_err, raw_err)

    def test_never_worse_than_translation_only(self) -> None:
        # The similarity family contains the translation-only candidate, so the
        # LS optimum can never have a larger sum of squared errors.
        rng = np.random.default_rng(1)
        for _ in range(10):
            gt = rng.normal(size=(15, 3))
            src = rng.normal(size=(15, 3))
            aligned = procrustes_align(src, gt)
            translated = src - src.mean(0, keepdims=True) + gt.mean(0, keepdims=True)
            self.assertLessEqual(
                float(np.sum((aligned - gt) ** 2)),
                float(np.sum((translated - gt) ** 2)) + 1e-8,
            )

    def test_degenerate_source_is_finite(self) -> None:
        gt = np.random.default_rng(2).normal(size=(10, 3))
        src = np.full((10, 3), 0.5)
        aligned = procrustes_align(src, gt)
        self.assertTrue(np.isfinite(aligned).all())

    def test_torch_variant_matches_numpy(self) -> None:
        from TriPoseFusion.eval.eval_trifusion_pesudo_gt import _procrustes_align

        rng = np.random.default_rng(3)
        gt = rng.normal(size=(12, 3))
        src = 0.8 * (gt @ _random_rotation(rng).T) + 0.3 + 0.02 * rng.normal(size=gt.shape)
        expected = procrustes_align(src, gt)
        actual = _procrustes_align(
            torch.from_numpy(src).double(), torch.from_numpy(gt).double()
        ).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class CanonicalizeFallbackTests(unittest.TestCase):
    J = 52

    def _pose(self, rng: np.random.Generator, frames: int) -> np.ndarray:
        pose = rng.normal(scale=0.3, size=(frames, self.J, 3)).astype(np.float32)
        pose[:, 5] = [-0.2, 0.0, 0.0]  # left shoulder
        pose[:, 6] = [0.2, 0.0, 0.0]  # right shoulder
        pose[:, 51] = [0.0, -0.1, 0.0]  # neck
        return pose

    def test_numpy_degenerate_frame_uses_neighbor_axis(self) -> None:
        rng = np.random.default_rng(4)
        pose = self._pose(rng, frames=5)
        pose[2, 5] = pose[2, 6] = [0.1, 0.1, 0.1]  # coincident shoulders
        out = canonicalize_pose(pose)
        self.assertTrue(np.isfinite(out).all())
        # The old fallback collapsed the whole frame to the origin.
        self.assertGreater(float(np.linalg.norm(out[2])), 1e-3)

    def test_numpy_all_invalid_sequence_stays_finite(self) -> None:
        rng = np.random.default_rng(5)
        pose = self._pose(rng, frames=3)
        pose[:, 5] = pose[:, 6] = 0.0
        out = canonicalize_pose(pose)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(np.linalg.norm(out)), 1e-3)

    def test_torch_fallback_preserves_norm(self) -> None:
        rng = np.random.default_rng(6)
        pose = torch.from_numpy(self._pose(rng, frames=4)).unsqueeze(0)
        pose[0, 2, 5] = pose[0, 2, 6] = torch.tensor([0.05, 0.05, 0.05])
        neck = pose[:, :, 51:52]
        left = pose[:, :, 5:6]
        right = pose[:, :, 6:7]
        canon = RobustCanonicalization()
        out = canon(pose, neck, left, right)
        self.assertTrue(bool(torch.isfinite(out).all()))
        # A rotation preserves per-frame norms of the neck-centered pose; the
        # old zero-rotation fallback mapped the degenerate frame to zero.
        centered = (pose - neck)[0, 2]
        self.assertAlmostEqual(
            float(torch.linalg.norm(out[0, 2])),
            float(torch.linalg.norm(centered)),
            places=4,
        )


class SameMaskMetricTests(unittest.TestCase):
    def test_compute_metrics_reports_same_mask_mpjpe(self) -> None:
        rng = np.random.default_rng(7)
        gt = rng.normal(size=(6, 10, 3))
        pred = gt + 0.05 * rng.normal(size=gt.shape)
        valid = np.ones((6, 10), dtype=bool)
        metrics = compute_metrics(pred, gt, valid)
        self.assertIn("mpjpe_pa_frames_m", metrics)
        # With every frame PA-eligible the same-mask MPJPE equals plain MPJPE,
        # and the corrected PA-MPJPE cannot exceed it.
        self.assertAlmostEqual(
            metrics["mpjpe_pa_frames_m"], metrics["mpjpe_m"], places=8
        )
        self.assertLessEqual(metrics["pa_mpjpe_m"], metrics["mpjpe_m"] + 1e-8)


if __name__ == "__main__":
    unittest.main()
