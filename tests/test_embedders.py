"""Ollama 日志向量化器测试（mock HTTP，不依赖真实 Ollama 服务）。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from aigc_core import GameLogRecord, OllamaLogEmbedder


class OllamaLogEmbedderTests(unittest.TestCase):
    def _record(self) -> GameLogRecord:
        return GameLogRecord(
            event_id="npc.1",
            event_type="npc.dialogue_line",
            data={"text": "你好"},
        )

    @patch("httpx.post")
    def test_embed_text_uses_api_embed(self, mock_post: Mock) -> None:
        mock_resp = Mock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_post.return_value = mock_resp

        embedder = OllamaLogEmbedder(model="qwen3-embedding:0.6b", base_url="http://ollama:11434")
        vector = embedder(self._record())

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        mock_post.assert_called_once()
        args = mock_post.call_args
        self.assertEqual(args[0][0], "http://ollama:11434/api/embed")

    @patch("httpx.post")
    def test_embed_text_falls_back_to_old_api(self, mock_post: Mock) -> None:
        first_resp = Mock()
        first_resp.raise_for_status.side_effect = RuntimeError("new api unavailable")
        second_resp = Mock()
        second_resp.json.return_value = {"embedding": [0.5, 0.6]}
        mock_post.side_effect = [first_resp, second_resp]

        embedder = OllamaLogEmbedder(model="mxbai-embed-large", base_url="http://ollama:11434")
        vector = embedder(self._record())

        self.assertEqual(vector, [0.5, 0.6])
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_post.call_args_list[1].args[0], "http://ollama:11434/api/embeddings")


if __name__ == "__main__":
    unittest.main()
