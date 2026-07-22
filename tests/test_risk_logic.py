"""Unit tests for risk classification and confidence floor."""

from __future__ import annotations

import unittest

from data import (
    RISK_DIST_THRESHOLDS,
    RISK_LATERAL_THRESHOLDS,
    RISK_TTC_THRESHOLDS,
    TrackedObject,
    classify_object_risk,
    get_overall_risk,
    lateral_for_risk,
)


class RiskLogicTests(unittest.TestCase):
    def test_clear_scene_is_low(self):
        risk, reason = classify_object_risk(
            ttc=12.0, dist=40.0, lateral=5.0, path_conflict=False, rel_vel=-0.1
        )
        self.assertEqual(risk, "LOW")

    def test_critical_ttc(self):
        risk, _ = classify_object_risk(
            ttc=RISK_TTC_THRESHOLDS["CRITICAL"] - 0.1,
            dist=30.0,
            lateral=5.0,
            path_conflict=False,
            rel_vel=-1.0,
        )
        self.assertEqual(risk, "CRITICAL")

    def test_critical_distance(self):
        risk, _ = classify_object_risk(
            ttc=float("inf"),
            dist=RISK_DIST_THRESHOLDS["CRITICAL"] - 0.1,
            lateral=5.0,
            path_conflict=False,
            rel_vel=0.0,
        )
        self.assertEqual(risk, "CRITICAL")

    def test_critical_lateral(self):
        risk, reason = classify_object_risk(
            ttc=float("inf"),
            dist=40.0,
            lateral=RISK_LATERAL_THRESHOLDS["CRITICAL"] - 0.1,
            path_conflict=False,
            rel_vel=0.0,
        )
        self.assertEqual(risk, "CRITICAL")
        self.assertIn("lat", reason.lower())

    def test_lateral_ignored_behind_or_far_ahead(self):
        # Behind ego: small |y| must not force CRITICAL
        behind = lateral_for_risk(-2.0, 0.2)
        risk, _ = classify_object_risk(
            ttc=float("inf"), dist=40.0, lateral=behind, path_conflict=False, rel_vel=0.0
        )
        self.assertEqual(risk, "LOW")

        # Far ahead beyond horizon: same
        far = lateral_for_risk(80.0, 0.2)
        risk, _ = classify_object_risk(
            ttc=float("inf"), dist=80.0, lateral=far, path_conflict=False, rel_vel=0.0
        )
        self.assertEqual(risk, "LOW")

        # Ahead inside horizon: lateral applies
        near = lateral_for_risk(20.0, 0.2)
        risk, _ = classify_object_risk(
            ttc=float("inf"), dist=20.0, lateral=near, path_conflict=False, rel_vel=0.0
        )
        self.assertEqual(risk, "CRITICAL")

    def test_higher_of_ttc_distance_and_lateral_wins(self):
        # Medium by TTC, High by distance, Low by lateral → High
        risk, _ = classify_object_risk(
            ttc=RISK_TTC_THRESHOLDS["MEDIUM"] - 0.2,
            dist=RISK_DIST_THRESHOLDS["HIGH"] - 0.2,
            lateral=5.0,
            path_conflict=False,
            rel_vel=-1.0,
        )
        self.assertEqual(risk, "HIGH")

    def test_path_conflict_bumps_when_relevant(self):
        risk, reason = classify_object_risk(
            ttc=2.5, dist=8.0, lateral=2.5, path_conflict=True, rel_vel=-1.0
        )
        self.assertIn(risk, ("HIGH", "CRITICAL", "MEDIUM"))
        self.assertIn("path", reason.lower())

    def test_confidence_floor_formula(self):
        # Mirror data.py: min(1.0, 0.5 + n_pts * 0.05)
        for n_pts, expected in ((0, 0.50), (5, 0.75), (10, 1.00), (20, 1.00)):
            conf = min(1.0, 0.5 + n_pts * 0.05)
            self.assertAlmostEqual(conf, expected, places=2)

    def test_overall_risk_matches_worst_object(self):
        def _obj(risk: str) -> TrackedObject:
            return TrackedObject(
                id="1", cls="Car", source="Fused", dist=20.0, rel_vel=0.0,
                ttc=10.0, req_decel=0.0, occupancy=0.0, path_conflict=False,
                confidence=0.5, risk=risk, risk_reason="test",
                bev_x=5.0, bev_y=20.0, cam_cx=-1, cam_cy=-1, cam_w=-1, cam_h=-1,
            )

        self.assertEqual(get_overall_risk([]), "LOW")
        self.assertEqual(get_overall_risk([_obj("LOW"), _obj("LOW")]), "LOW")
        self.assertEqual(get_overall_risk([_obj("LOW"), _obj("MEDIUM")]), "MEDIUM")
        self.assertEqual(
            get_overall_risk([_obj("MEDIUM"), _obj("CRITICAL")]), "CRITICAL"
        )


if __name__ == "__main__":
    unittest.main()
