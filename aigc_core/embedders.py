"""向量化器：把日志/文本变成 semantic_vector。

当前先提供 Ollama 本地 embedding 适配器，默认 `bge-m3` 已于 2026-09-06切换为 `qwen3-embedding:0.6b`（A/B 判别 8/10 vs 5/10，见 aigc_docs/13 §7）。
不接外部服务；只调用本地 Ollama HTTP 接口。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import GameLogRecord


def _embed_keep_alive():
    """embedding 常驻位策略(动态批处理管理, 2026-09-07): 管理器不在路径上时退回 30m。"""
    try:
        from model_batch_manager import EMBED_MODEL, policy_keep_alive
        return policy_keep_alive(EMBED_MODEL)
    except Exception:
        return "30m"


class OllamaLogEmbedder:
    """调用本地 Ollama 的 embedding 模型给 Godot 日志生成向量。

    支持新版 `POST /api/embed`，并兼容旧版 `POST /api/embeddings`。
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_text(self, text: str) -> List[float]:
        import httpx

        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text, "keep_alive": _embed_keep_alive()},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
            embeddings = data.get("embeddings") or data.get("data")
            if embeddings:
                return [float(x) for x in embeddings[0]]
        except Exception:
            pass

        # 旧版 Ollama API 兼容
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding") or data.get("embeddings", [None])[0]
        if not embedding:
            raise RuntimeError(f"ollama embed returned no vector: {data}")
        return [float(x) for x in embedding]

    def __call__(self, record: GameLogRecord) -> List[float]:
        return self.embed_text(record.to_embedding_text())

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入: 一次 /api/embed 请求携带多条输入(实测比逐条快 ~15 倍)。

        任一批内失败整体抛出, 由调用方决定降级策略。
        """
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts, "keep_alive": _embed_keep_alive()},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()
        embeddings = data.get("embeddings") or data.get("data")
        if not embeddings or len(embeddings) != len(texts):
            got = len(embeddings) if embeddings else 0
            raise RuntimeError(f"ollama batch embed mismatch: got {got}, want {len(texts)}")
        return [[float(x) for x in vec] for vec in embeddings]


_EMBED_CB = {"fail_until": 0.0, "fails": 0}


def embed_text_best_effort(
    text: str,
    *,
    model: str = "qwen3-embedding:0.6b",
    base_url: str = "http://localhost:11434",
    timeout: float = 4.0,
) -> List[float]:
    """/ctx 通路的降级嵌入：失败返回 []（词法通道兜底），连续失败后熔断 60s。

    供 controller.context_ingest / api /ctx/recall 使用——嵌入不可用绝不阻塞
    写入与召回主路径，只是该条/该次退化为 lexical-only。
    """
    import time

    now = time.time()
    if now < _EMBED_CB["fail_until"]:
        return []
    try:
        vec = OllamaLogEmbedder(model=model, base_url=base_url, timeout=timeout).embed_text(text)
        _EMBED_CB["fails"] = 0
        return vec
    except Exception:
        _EMBED_CB["fails"] += 1
        if _EMBED_CB["fails"] >= 2:
            _EMBED_CB["fail_until"] = now + 60.0
            _EMBED_CB["fails"] = 0
        return []


class QueryEmbedCache:
    """查询嵌入缓存: 交互式检索里同一查询文本反复出现, 二次起零成本。"""

    def __init__(self, embedder, cap: int = 256) -> None:
        self._embedder = embedder
        self._cap = cap
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        vec = self._cache.get(text)
        if vec is None:
            vec = self._embedder.embed_text(text)
            if len(self._cache) >= self._cap:
                self._cache.clear()
            self._cache[text] = vec
        return vec

    def __len__(self) -> int:
        return len(self._cache)
