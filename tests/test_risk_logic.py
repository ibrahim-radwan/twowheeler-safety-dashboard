"""Unit tests for risk classification and confidence floor."""

from __future__ import annotations

import unittest

from data import (
    RISK_DIST_THRESHOLDS,
    RISK_TTC_THRESHOLDS,
    classify_object_risk,
)


class RiskLogicTests(unittest.TestCase):
    def test_clear_scene_is_low(self):
        risk, reason = classify_object_risk(
            ttc=12.0, dist=40.0, path_conflict=False, rel_vel=-0.1
        )
        self.assertEqual(risk, "LOW")

    def test_critical_ttc(self):
        risk, _ = classify_object_risk(
            ttc=RISK_TTC_THRESHOLDS["CRITICAL"] - 0.1,
            dist=30.0,
            path_conflict=False,
            rel_vel=-1.0,
        )
        self.assertEqual(risk, "CRITICAL")

    def test_critical_distance(self):
        risk, _ = classify_object_risk(
            ttc=float("inf"),
            dist=RISK_DIST_THRESHOLDS["CRITICAL"] - 0.1,
            path_conflict=False,
            rel_vel=0.0,
        )
        self.assertEqual(risk, "CRITICAL")

    def test_higher_of_ttc_and_distance_wins(self):
        # Medium by TTC, High by distance → High
        risk, _ = classify_object_risk(
            ttc=RISK_TTC_THRESHOLDS["MEDIUM"] - 0.2,
            dist=RISK_DIST_THRESHOLDS["HIGH"] - 0.2,
            path_conflict=False,
            rel_vel=-1.0,
        )
        self.assertEqual(risk, "HIGH")

    def test_path_conflict_bumps_when_relevant(self):
        risk, reason = classify_object_risk(
            ttc=2.5, dist=8.0, path_conflict=True, rel_vel=-1.0
        )
        self.assertIn(risk, ("HIGH", "CRITICAL", "MEDIUM"))
        self.assertIn("path", reason.lower())

    def test_confidence_floor_formula(self):
        # Mirror data.py: min(1.0, 0.5 + n_pts * 0.05)
        for n_pts, expected in ((0, 0.50), (5, 0.75), (10, 1.00), (20, 1.00)):
            conf = min(1.0, 0.5 + n_pts * 0.05)
            self.assertAlmostEqual(conf, expected, places=2)


if __name__ == "__main__":
    unittest.main()
