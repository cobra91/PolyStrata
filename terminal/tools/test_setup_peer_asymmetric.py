"""Asymmetric dual-GPU detection and stronger-primary selection for --peer-device.

    python -m unittest tools.test_setup_peer_asymmetric
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class AsymmetricDualGpu(unittest.TestCase):
    def test_rtx5080_plus_rtx4060ti_detects_asymmetry(self):
        rtx5080 = {"index": 0, "name": "NVIDIA GeForce RTX 5080", "arch": 120, "vram_gb": 16.0}
        rtx4060ti = {"index": 1, "name": "NVIDIA GeForce RTX 4060 Ti", "arch": 89, "vram_gb": 16.0}
        link5080 = {"gen": 5, "gpu_gen": 5, "host_gen": 5, "width": 16, "max_width": 16}
        link4060ti = {"gen": 4, "gpu_gen": 4, "host_gen": 4, "width": 2, "max_width": 8}

        info = setup.asymmetric_dual_gpu_info(rtx5080, rtx4060ti, link5080, link4060ti)
        self.assertTrue(info["is_asymmetric"])
        self.assertEqual(info["stronger_index"], 0)
        self.assertEqual(info["recommended_env"], setup.ASYMMETRIC_PEER_ENV)
        joined = " ".join(info["reasons"])
        self.assertIn("sm_120 vs sm_89", joined)
        self.assertIn("PCIe 5.0 x16", joined)
        self.assertTrue(any("STRATA_ROUTE_TAIL_SKIP=7" in n for n in info["optional_notes"]))
        self.assertTrue(any("--ram-keep-gib 8" in n for n in info["optional_notes"]))

    def test_reversed_order_identifies_stronger_primary(self):
        rtx5080 = {"index": 0, "name": "NVIDIA GeForce RTX 5080", "arch": 120, "vram_gb": 16.0}
        rtx4060ti = {"index": 1, "name": "NVIDIA GeForce RTX 4060 Ti", "arch": 89, "vram_gb": 16.0}
        link5080 = {"gen": 5, "gpu_gen": 5, "host_gen": 5, "width": 16, "max_width": 16}
        link4060ti = {"gen": 4, "gpu_gen": 4, "host_gen": 4, "width": 2, "max_width": 8}

        info = setup.asymmetric_dual_gpu_info(rtx4060ti, rtx5080, link4060ti, link5080)
        self.assertTrue(info["is_asymmetric"])
        self.assertEqual(info["stronger_index"], 0)

    def test_same_arch_vram_pcie_asymmetry_picks_faster_link(self):
        card0 = {"index": 0, "name": "RTX 4090", "arch": 89, "vram_gb": 24.0}
        card1 = {"index": 1, "name": "RTX 4090", "arch": 89, "vram_gb": 24.0}
        slow = {"gen": 3, "gpu_gen": 4, "host_gen": 3, "width": 4, "max_width": 16}
        fast = {"gen": 4, "gpu_gen": 4, "host_gen": 4, "width": 16, "max_width": 16}

        info = setup.asymmetric_dual_gpu_info(card0, card1, slow, fast)
        self.assertTrue(info["is_asymmetric"])
        self.assertEqual(info["stronger_index"], 1)

    def test_symmetric_pair_returns_no_override(self):
        card0 = {"index": 0, "name": "RTX 4090", "arch": 89, "vram_gb": 24.0}
        card1 = {"index": 1, "name": "RTX 4090", "arch": 89, "vram_gb": 24.0}
        link = {"gen": 4, "gpu_gen": 4, "host_gen": 4, "width": 16, "max_width": 16}

        info = setup.asymmetric_dual_gpu_info(card0, card1, link, link)
        self.assertFalse(info["is_asymmetric"])
        self.assertEqual(info["reasons"], [])
        self.assertEqual(info["recommended_env"], {})
        self.assertEqual(info["optional_notes"], [])


if __name__ == "__main__":
    unittest.main()
