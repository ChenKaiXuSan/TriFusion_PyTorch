from pathlib import Path
import tempfile
import unittest

from TriPoseFusion.eval.eval_ablation_logs_pseudo_gt import (
    build_eval_command,
    discover_runs,
    select_checkpoint,
)


def _make_run(root: Path, name: str) -> Path:
    run_dir = root / name / "2026-06-11" / "10-00-00"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra" / "config.yaml").write_text("experiment: demo\n")
    ckpt_dir = run_dir / "checkpoints" / "fold_0"
    ckpt_dir.mkdir(parents=True)
    return run_dir


class EvalAblationLogsPseudoGtTests(unittest.TestCase):
    def test_discover_runs_finds_hydra_checkpoint_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            (run_dir / "checkpoints" / "fold_0" / "last.ckpt").write_text("x")
            (tmp_path / "not_a_run").mkdir()

            runs = discover_runs(tmp_path, pattern="trifusion_*", fold=0)

            self.assertEqual(runs, [run_dir])

    def test_select_checkpoint_best_uses_lowest_metric_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            ckpt_dir = run_dir / "checkpoints" / "fold_0"
            (ckpt_dir / "last.ckpt").write_text("last")
            (ckpt_dir / "2-0.87.ckpt").write_text("worse")
            best = ckpt_dir / "0-0.86.ckpt"
            best.write_text("best")

            self.assertEqual(select_checkpoint(run_dir, fold=0, policy="best"), best)

    def test_select_checkpoint_last_uses_last_ckpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            ckpt_dir = run_dir / "checkpoints" / "fold_0"
            (ckpt_dir / "0-0.86.ckpt").write_text("best")
            last = ckpt_dir / "last.ckpt"
            last.write_text("last")

            self.assertEqual(select_checkpoint(run_dir, fold=0, policy="last"), last)

    def test_build_eval_command_uses_run_config_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = _make_run(tmp_path, "trifusion_base")
            ckpt = run_dir / "checkpoints" / "fold_0" / "0-0.86.ckpt"
            output_dir = tmp_path / "eval_out" / "trifusion_base"
            gt_root = tmp_path / "gt"
            eval_script = tmp_path / "eval_trifusion_pesudo_gt.py"

            command = build_eval_command(
                python_executable="python",
                eval_script=eval_script,
                run_dir=run_dir,
                ckpt_path=ckpt,
                output_dir=output_dir,
                gt_root=gt_root,
                split="val",
                fold="0",
                extra_overrides=["data.num_workers=0"],
            )

            self.assertEqual(
                command[:5],
                [
                    "python",
                    str(eval_script),
                    "--config-path",
                    str(run_dir / ".hydra"),
                    "--config-name",
                ],
            )
            self.assertIn("config", command)
            self.assertIn(f"eval.ckpt_path={ckpt}", command)
            self.assertIn(f"eval.output_dir={output_dir}", command)
            self.assertIn(f"eval.triangulated_gt_root={gt_root}", command)
            self.assertIn("eval.split=val", command)
            self.assertIn("eval.fold=0", command)
            self.assertIn("data.num_workers=0", command)


if __name__ == "__main__":
    unittest.main()
