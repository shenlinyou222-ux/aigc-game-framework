"""接口向量库测试：接口定义 -> 向量 -> 语义召回 -> JSONL 持久化。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aigc_core.interface_store import (
    HashEmbedder,
    InterfaceSpec,
    InterfaceVectorStore,
    default_interface_specs,
)


class InterfaceVectorStoreTests(unittest.TestCase):
    def test_default_specs_include_core_commands(self) -> None:
        names = {s.name for s in default_interface_specs()}
        self.assertIn("play_music", names)
        self.assertIn("create_item", names)
        self.assertIn("give_item", names)
        self.assertIn("query_logs", names)

    def test_add_query_returns_top_hit(self) -> None:
        store = InterfaceVectorStore()
        store.add_many(default_interface_specs())
        self.assertGreater(store.count(), 0)

        # 创建意图 → create_item 应排前列
        hits = store.query("创建一种叫 test 的新物品种类", top_k=3)
        self.assertIn("create_item", [h.spec.name for h in hits])
        # 发放意图 → give_item 应排前列
        hits = store.query("给玩家发放一个叫 test 的物品", top_k=3)
        self.assertIn("give_item", [h.spec.name for h in hits])

    def test_query_music(self) -> None:
        store = InterfaceVectorStore()
        store.add_many(default_interface_specs())
        hits = store.query("换一首轻松的背景音乐", top_k=1)
        self.assertEqual(hits[0].spec.name, "play_music")

    def test_hash_embedder_is_deterministic(self) -> None:
        emb = HashEmbedder(dim=32)
        a = emb("play_music switch background music")
        b = emb("play_music switch background music")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)

    def test_persist_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interfaces.jsonl"
            store = InterfaceVectorStore(path=path)
            store.add_many(default_interface_specs())

            reloaded = InterfaceVectorStore(path=path)
            self.assertEqual(reloaded.count(), store.count())
            self.assertEqual(reloaded.get("play_music").name, "play_music")
            self.assertEqual(
                reloaded.query("换一首轻松的背景音乐", top_k=1)[0].spec.name,
                "play_music",
            )


if __name__ == "__main__":
    unittest.main()
