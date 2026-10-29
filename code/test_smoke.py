"""Small, deterministic implementation checks; does not start a research run."""
import tempfile
import unittest
from pathlib import Path
import sys
import types
import numpy as np
import torch
from scipy.io import savemat

# Import this folder under a unique package name to avoid shadowing stdlib `code`.
egu_simple = types.ModuleType("egu_simple")
egu_simple.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["egu_simple"] = egu_simple
from egu_simple.data_loader import load_samson
from egu_simple.egu_net_ss import EGUNetSS
from egu_simple.losses import total_loss
from egu_simple.checkpoint import save_checkpoint, load_checkpoint
from egu_simple.evaluation import metrics, abundance_maps_to_pixels
from egu_simple.visualization import save_evaluation_figures


class SmokeTests(unittest.TestCase):
    def test_matlab_loader_dimensions_order(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); h, w, b, c = 4, 3, 7, 2
            cube = np.arange(h*w*b, dtype=np.float32).reshape(h, w, b)
            v = cube.transpose(2, 0, 1).reshape(b, -1, order="F")
            a = np.full((c, h*w), 1/c, np.float32)
            m = np.ones((b, c), np.float32)
            savemat(p/"hsi.mat", {"V": v})
            savemat(p/"gt.mat", {"A": a, "M": m})
            loaded = load_samson(p/"hsi.mat", p/"gt.mat", (h, w, b), c)
            self.assertEqual(loaded["cube"].shape, (h, w, b))
            np.testing.assert_allclose(loaded["spectra"], v)
            self.assertEqual(loaded["abundances_gt"].shape, (c, h*w))

    def test_model_losses_gradients_sharing(self):
        torch.manual_seed(2)
        model = EGUNetSS(bands=8, materials=2, dropout_rate=0.1)
        self.assertTrue(model.sharing_is_valid())
        pure = torch.rand(6, 8)
        cube = torch.rand(1, 8, 17, 19)
        labels = torch.softmax(torch.rand(6, 2), dim=1)
        result = model(pure, cube)
        self.assertEqual(tuple(result["abundances"].shape), (1, 2, 17, 19))
        self.assertEqual(tuple(result["reconstruction"].shape), tuple(cube.shape))
        abundance = result["abundances"]
        self.assertGreaterEqual(float(abundance.min()), 0)
        torch.testing.assert_close(abundance.sum(1), torch.ones_like(abundance[:, 0]))
        loss, cls, rec, mix, abundance_loss, reg = total_loss(
            result["class_logits"], labels, result["reconstruction"], cube, model
        )
        self.assertTrue(
            all(torch.isfinite(x) for x in
                (loss, cls, rec, mix, abundance_loss, reg))
        )
        loss.backward()
        self.assertIsNotNone(model.enet.shared_conv3.weight.grad)
        self.assertIsNotNone(model.urnet.spatial_conv1.weight.grad)

    def test_checkpoint_resume_and_metrics_figures(self):
        model = EGUNetSS(bands=8, materials=2, dropout_rate=0.1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        cfg = {"bands": 8, "materials": 2, "dropout_rate": 0.1}
        state = {"epoch": 1, "batch_cursor": 3, "global_step": 4,
                 "order": [2, 0, 1], "partial_sums": {"total": 0.4},
                 "history": [{"epoch": 1, "total": 0.2, "classification": 0.1, "reconstruction": 0.1}]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"latest.pt"
            save_checkpoint(path, model, opt, sched, state, cfg)
            loaded = load_checkpoint(path, model, opt, sched, cfg)
            self.assertEqual(loaded["state"]["batch_cursor"], 3)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, expected_config={"bands": 9, "materials": 2, "dropout_rate": .1})
            maps = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            pixels = abundance_maps_to_pixels(maps, 2, 3)
            np.testing.assert_array_equal(pixels[0], [0, 3, 1, 4, 2, 5])
            gt = np.array([[1, 0, .5, 0], [0, 1, .5, 1]], np.float32)
            pred = gt[::-1].copy()
            ref_e = np.eye(8, 2, dtype=np.float32)
            out = metrics(gt, pred, ref_e, ref_e[:, ::-1])
            self.assertEqual(out[0]["predicted_channel_order"], [1, 0])
            self.assertAlmostEqual(out[0]["aRMSE"], 0.0, places=6)
            history = [{"total": 1., "classification": .5, "reconstruction": .5}]
            paths = save_evaluation_figures(gt, gt.reshape(2, 2, 2), ref_e,
                                            ref_e, history, (2, 2, 8), ["a", "b"], Path(td)/"figs")
            self.assertTrue(all(p.is_file() and p.stat().st_size > 0 for p in paths))


if __name__ == "__main__":
    unittest.main()
