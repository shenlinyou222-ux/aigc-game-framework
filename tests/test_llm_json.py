# -*- coding: utf-8 -*-
"""llm_json 单测：三层修复链 + pydantic 矫正 + generate_json 假传输。全部离线，不碰真网络。"""
import unittest

from aigc_core.llm_json import (
    DialogueReply,
    InferenceVerdict,
    adapt_dialogue,
    adapt_inference,
    generate_json,
    repair_json,
)
from model_batch_manager import policy_keep_alive


class TestRepairJson(unittest.TestCase):
    def test_real_world_bare_key(self):
        # npc_scene 实测抓到的 0.8b 原始坏输出
        obj, r = repair_json('{"attitude": "敌意", "new_memory": "老鼠B判定玩家杀害了同伴A", ,confidence: 0.85}')
        self.assertEqual(obj["confidence"], 0.85)
        self.assertEqual(obj["attitude"], "敌意")
        self.assertIn("json_repair", r)

    def test_fence_and_noise(self):
        obj, _ = repair_json('好的，以下是JSON：\n```json\n{"attitude": "仇恨"}\n```\n希望有帮助')
        self.assertEqual(obj, {"attitude": "仇恨"})

    def test_single_quote_trailing_comma(self):
        obj, _ = repair_json("{'attitude': '恐惧', 'confidence': 0.88,}")
        self.assertEqual(obj["confidence"], 0.88)

    def test_truncated(self):
        obj, _ = repair_json('{"attitude": "敌意", "confidence": 0.7')
        self.assertEqual(obj["confidence"], 0.7)

    def test_think_leak(self):
        obj, r = repair_json('<think>\n推理过程...\n</think>\n{"attitude": "仇恨"}')
        self.assertEqual(obj["attitude"], "仇恨")
        self.assertIn("strip_think", r)

    def test_fullwidth_structural(self):
        obj, r = repair_json('{"attitude"："仇恨"，"confidence"：0.9}')
        self.assertEqual(obj["attitude"], "仇恨")
        self.assertEqual(obj["confidence"], 0.9)
        self.assertIn("fullwidth_punct", r)

    def test_fullwidth_inside_string_preserved(self):
        # 台词值里的中文标点是内容，不许动
        obj, _ = repair_json('{"new_memory": "老鼠B说：你好，玩家。"}')
        self.assertEqual(obj["new_memory"], "老鼠B说：你好，玩家。")

    def test_unrepairable(self):
        obj, r = repair_json("我拒绝回答")
        self.assertIsNone(obj)
        self.assertIn("unrepairable", r)

    def test_clean_json_zero_repairs(self):
        obj, r = repair_json('{"attitude": "敌意"}')
        self.assertEqual(r, [])


class TestInferenceAdapt(unittest.TestCase):
    def test_confidence_string_coerce(self):
        v, _ = adapt_inference({"attitude": "仇恨", "new_memory": "老鼠B仇恨玩家", "confidence": "0.75"})
        self.assertEqual(v.confidence, 0.75)

    def test_confidence_percent_number(self):
        v, _ = adapt_inference({"attitude": "仇恨", "new_memory": "x", "confidence": 90})
        self.assertEqual(v.confidence, 0.9)

    def test_confidence_missing_none(self):
        # 缺 confidence → None，由调用方落人物数据默认值（现有 S4 语义）
        v, _ = adapt_inference({"attitude": "仇恨", "new_memory": "x"})
        self.assertIsNone(v.confidence)

    def test_confidence_out_of_range_fail_closed(self):
        v, _ = adapt_inference({"attitude": "仇恨", "new_memory": "x", "confidence": "abc"})
        self.assertIsNone(v.confidence)


class TestDialogueAdapt(unittest.TestCase):
    def test_lines_string_to_list(self):
        d, _ = adapt_dialogue({"lines": "哼，才不是在等你；摸头也不是不行"})
        self.assertEqual(len(d.lines), 2)
        self.assertTrue(all(isinstance(x, str) for x in d.lines))

    def test_new_memory_string_to_object(self):
        d, _ = adapt_dialogue({"lines": ["哼"], "new_memory": "老鼠B记住了这件事"})
        self.assertEqual(d.new_memory.text, "老鼠B记住了这件事")
        self.assertIsNone(d.new_memory.grounded)

    def test_trigger_default(self):
        d, _ = adapt_dialogue({"lines": ["哼"]})
        self.assertEqual(d.trigger, {"type": "proximity", "radius_px": 64})

    def test_full_dialogue(self):
        d, _ = adapt_dialogue({
            "lines": ["站住！", "本喵等你好久了喵"],
            "emotion": "傲娇",
            "trigger": {"type": "proximity", "radius_px": 64},
            "claims": [{"text": "上次暴雨你救了我", "event_type": "story_dialog.finished", "day": 3}],
            "new_memory": {"text": "玩家救过我", "grounded": True, "confidence": 0.8},
        })
        self.assertEqual(len(d.claims), 1)
        self.assertEqual(d.new_memory.confidence, 0.8)


class TestGenerateJson(unittest.TestCase):
    """假传输：验证 format/keep_alive/num_ctx 请求形状 + 修复链贯通。"""

    def _fake_transport(self, captured, raw):
        def t(body):
            captured.append(body)
            return {"response": raw}
        return t

    def test_request_shape(self):
        captured = []
        t = self._fake_transport(captured, '{"attitude": "仇恨", "new_memory": "x"}')
        v, audit = generate_json("p", adapter=adapt_inference, json_schema={"type": "object"},
                                 transport=t)
        body = captured[0]
        self.assertEqual(body["keep_alive"], policy_keep_alive())  # 2026-09-07 起非常驻: 动态批处理策略(默认45s空闲窗)
        self.assertEqual(body["options"]["num_ctx"], 2048)    # NPC 对话小上下文
        self.assertEqual(body["think"], False)
        self.assertEqual(body["format"], {"type": "object"})  # 结构化输出
        self.assertEqual(v.attitude, "仇恨")
        self.assertEqual(audit["repairs"], [])

    def test_repair_pipeline_through_generate(self):
        captured = []
        t = self._fake_transport(captured, '{"attitude": "敌意", ,confidence: 0.85}')
        v, audit = generate_json("p", adapter=adapt_inference, transport=t)
        self.assertEqual(v.confidence, 0.85)
        self.assertIn("json_repair", audit["repairs"])

    def test_unrepairable_returns_none(self):
        t = self._fake_transport([], "我拒绝回答")
        v, audit = generate_json("p", adapter=adapt_inference, transport=t)
        self.assertIsNone(v)
        self.assertIn("unrepairable", audit["repairs"])

    def test_model_validate_error_propagates(self):
        t = self._fake_transport([], "12345")   # 合法 JSON 但不是对象
        v, audit = generate_json("p", adapter=adapt_inference, transport=t)
        self.assertIsNone(v)
        self.assertIn("adapt_error", audit)


if __name__ == "__main__":
    unittest.main()
