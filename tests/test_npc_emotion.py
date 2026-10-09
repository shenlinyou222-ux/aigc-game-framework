# -*- coding: utf-8 -*-
"""Gamygdala 式情绪引擎测试：评价/归因/关系/衰减/态度词表。"""
import unittest

from aigc_core.npc_emotion import Agent, EmotionEngine, Goal, attitude_word


def _agent() -> Agent:
    return Agent("mouse_b", faction="mouse", goals=[
        Goal("同族安全", 1.0),
        Goal("自身生存", 0.8),
        Goal("囤粮过冬", 0.4, likelihood=0.8),
    ])


def _neg_event(target: str = "mouse_a") -> dict:
    return {"causal": "player", "target": target, "likelihood": 1.0,
            "goals": {"同族安全": -0.8}}


class TestAppraisal(unittest.TestCase):
    def test_negative_other_caused_makes_anger_and_drops_like(self):
        eng = EmotionEngine()
        ag = _agent()
        r = eng.appraise(ag, _neg_event(), "mouse_b")
        self.assertGreater(ag.emotions["distress"].intensity, 0)
        self.assertGreater(ag.emotions["anger"].intensity, 0)
        self.assertLess(ag.relation("player").like, 0)
        self.assertEqual(r["attitude"]["attitude"], "仇恨")

    def test_repeated_negative_builds_hate_with_slow_decay(self):
        eng = EmotionEngine()
        ag = _agent()
        for _ in range(2):
            eng.appraise(ag, _neg_event("mouse_b"), "mouse_b")
        self.assertIn("hate", ag.emotions)
        self.assertLess(ag.emotions["hate"].decay_factor, 1.0)  # 恨衰减更慢

    def test_positive_other_caused_makes_gratitude(self):
        eng = EmotionEngine()
        ag = _agent()
        eng.appraise(ag, {"causal": "player", "target": "mouse_b", "likelihood": 1.0,
                          "goals": {"囤粮过冬": 0.5}}, "mouse_b")
        self.assertGreater(ag.emotions["gratitude"].intensity, 0)
        self.assertGreater(ag.relation("player").like, 0)
        self.assertEqual(eng.attitude(ag, "player")["attitude"], "友善")

    def test_unrelated_event_no_emotion(self):
        eng = EmotionEngine()
        ag = _agent()
        r = eng.appraise(ag, {"causal": "player", "likelihood": 1.0,
                              "goals": {"不存在的目标": -0.5}}, "mouse_b")
        self.assertEqual(r["produced"], [])
        self.assertEqual(len(ag.emotions), 0)

    def test_intensity_formula_utility_delta_likelihood(self):
        eng = EmotionEngine()
        ag = _agent()
        eng.appraise(ag, {"causal": "storm", "likelihood": 0.5,
                          "goals": {"囤粮过冬": -1.0}}, "mouse_b")
        # |0.4 × -1.0| × 0.5 = 0.2（囤粮 likelihood=0.8 的 pending 分支→fear）
        self.assertAlmostEqual(ag.emotions["fear"].intensity, 0.2, places=3)


class TestDecayAndAttitude(unittest.TestCase):
    def test_decay_exponential_and_hate_slower(self):
        eng = EmotionEngine({"decay_per_day": 0.5, "hate_decay_factor": 0.4})
        ag = _agent()
        for _ in range(2):
            eng.appraise(ag, _neg_event("mouse_b"), "mouse_b")
        a0 = ag.emotions["anger"].intensity
        h0 = ag.emotions["hate"].intensity
        eng.decay(ag)
        self.assertAlmostEqual(ag.emotions["anger"].intensity, a0 * 0.5, places=3)
        self.assertAlmostEqual(ag.emotions["hate"].intensity, h0 * (1 - 0.5 * 0.4), places=3)

    def test_attitude_word_mapping(self):
        self.assertEqual(attitude_word(-0.8), "仇恨")
        self.assertEqual(attitude_word(-0.4), "敌意")
        self.assertEqual(attitude_word(0.0), "中性")
        self.assertEqual(attitude_word(0.4), "友善")
        self.assertEqual(attitude_word(0.9), "挚友")

    def test_engine_params_from_persona_data(self):
        eng = EmotionEngine.engine_from_persona(
            {"emotion_engine": {"decay_per_day": 0.9}})
        self.assertEqual(eng.config["decay_per_day"], 0.9)
        ag = _agent()
        eng.appraise(ag, _neg_event(), "mouse_b")
        eng.decay(ag)
        self.assertLess(ag.emotions["anger"].intensity, 0.2)  # 高衰减率快速平复


if __name__ == "__main__":
    unittest.main()
