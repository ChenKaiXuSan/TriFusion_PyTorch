import unittest
from types import SimpleNamespace

import numpy as np
import torch

from TriPoseFusion.eval.eval_additional_baselines_pseudo_gt import (
    best_single_payload,
    moving_average_pose,
)
from TriPoseFusion.models.keypoint_mlp import TriViewKeypointFusionNet, ZeroTemporalPoseRefiner


class AdditionalPoseBaselineTests(unittest.TestCase):
    def test_moving_average_pose_uses_centered_edge_padding(self):
        sequence = np.arange(5, dtype=np.float32).reshape(5, 1, 1)

        smoothed = moving_average_pose(sequence, window=3)

        expected = np.array([1 / 3, 1.0, 2.0, 3.0, 11 / 3], dtype=np.float32).reshape(5, 1, 1)
        np.testing.assert_allclose(smoothed, expected)

    def test_best_single_payload_selects_lowest_mpjpe(self):
        payloads = [
            {
                "method": "front_single",
                "source": "front",
                "metrics": {"mpjpe_m": 0.20},
                "cameras": {"front": {"num_frames": 4}},
            },
            {
                "method": "left_single",
                "source": "left",
                "metrics": {"mpjpe_m": 0.10},
                "cameras": {"left": {"num_frames": 4}},
            },
        ]

        best = best_single_payload(payloads)

        self.assertEqual(best["method"], "best_single")
        self.assertEqual(best["selected_single_method"], "left_single")
        self.assertEqual(best["metrics"]["mpjpe_m"], 0.10)

    def test_gate_only_config_uses_zero_temporal_refiner(self):
        hparams = SimpleNamespace(
            model=SimpleNamespace(
                geofusion_view_names=["front", "left", "right"],
                geofusion_num_joints=6,
                geofusion_hidden_dim=16,
                geofusion_refiner_dim=16,
                geofusion_refiner_layers=1,
                geofusion_dropout=0.0,
                geofusion_use_2d=False,
                geofusion_use_conf=False,
                geofusion_use_reproj_error_feature=False,
                geofusion_use_multiscale_velocity=False,
                geofusion_use_temporal_refiner=False,
                geofusion_use_cross_view_attention=False,
                geofusion_use_learned_gate=True,
                geofusion_canonicalize=False,
                geofusion_nce_dim=8,
                kpt_neck_index=0,
                kpt_left_shoulder_index=1,
                kpt_right_shoulder_index=2,
            )
        )
        model = TriViewKeypointFusionNet(hparams)

        pose3d = torch.randn(2, 3, 4, 6, 3)
        output = model(pose3d)

        self.assertIsInstance(model.refiner, ZeroTemporalPoseRefiner)
        self.assertTrue(torch.allclose(output["delta"], torch.zeros_like(output["delta"])))
        self.assertTrue(torch.allclose(output["P_final"], output["P_init"]))


if __name__ == "__main__":
    unittest.main()
