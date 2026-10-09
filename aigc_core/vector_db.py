"""向量数据库适配层：真正的向量存哪、生命周期/清理由谁管。

分层结论（重要）：
- 上下文向量应当存在**真正的向量数据库**里（ChromaDB / LanceDB / Qdrant / Milvus…）。
- **数据的持久化、删除、到期清理是数据库的职责**，应用层不做内存对象簿记/自己遍历删。
  我们的角色只有两个：
  1. 把生命周期策略翻译成一个可索引的 `expires_at`（= deadline_base + duration_sec）写进记录元数据；
  2. 对不支持原生 TTL 的库（如 Chroma），只发一次 `delete(where=expires_at<now)` 交给 DB 执行——
     我们绝不维护“该删谁的清单再逐个 pop”。

因此：
- `aigc_core/log_vector_store.py`（JSONL + 内存 dict）只是**离线/无库环境下的临时代替**，
  不是“向量数据库”，也不应被当作生命周期权威；
- 真库装上后，`LogVectorStore` 应整体被本文件的 `VectorDb` 实现替换。

本文件当前只做协议 + 一个可用的 ChromaDB 适配器（懒加载，未安装 Chroma 时不阻塞导入）。
Chroma 无原生 TTL，故由 `delete_expired()` 发一次 DB 删除；若换 Milvus/Qdrant 等带 TTL 的库，
则无需这个触发，库自己到期清除。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# 保留字段名常量：所有 DB 实现都用同一套，便于迁移。
EXPIRES_FIELD = "expires_at"
SESSION_FIELD = "session_id"
KIND_FIELD = "kind"


@dataclass(frozen=True)
class VectorDbConfig:
    """向量数据库连接配置（适配器专属字段由各自 __init__ 展开）。"""

    collection: str = "ctx_memory"
    persist_dir: str = "./vector_db"
    distance: str = "cosine"


class VectorHit(Protocol):
    event_id: str
    score: float
    record: Dict[str, Any]


class VectorDb(Protocol):
    """应用层只依赖这个端口，不依赖具体库。DB 拥有数据与清理。"""

    def add(
        self,
        event_id: str,
        vector: List[float],
        text: str,
        metadata: Dict[str, Any],
        expires_at: float,
    ) -> None: ...

    def search(
        self,
        vector: List[float],
        *,
        limit: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    def touch(self, event_id: str, expires_at: float) -> None: ...

    def delete_expired(self, now: Optional[float] = None) -> int: ...

    def delete(self, event_id: str) -> None: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class ChromaVectorDb:
    """ChromaDB 本地持久化实现（元数据扁平化，expires_at 进 where 条件）。

    Chroma 没有原生 TTL，因此到期删除走 `collection.delete(where=...)` 一次提交给 DB；
    应用层不持有内存对象清单。
    """

    def __init__(
        self,
        collection: str = "ctx_memory",
        persist_dir: str = "./vector_db",
        distance: str = "cosine",
    ) -> None:
        try:
            import chromadb  # 懒加载：没装 Chroma 时构造即抛，方便上层给出清晰报错
        except Exception as exc:  # pragma: no cover - 取决于是否安装
            raise RuntimeError(
                "chromadb is not installed; run: pip install chromadb  "
                "(或改用其它 VectorDb 实现，并让 LogVectorStore 退出生产路径)"
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": distance},
        )

    def add(
        self,
        event_id: str,
        vector: List[float],
        text: str,
        metadata: Dict[str, Any],
        expires_at: float,
    ) -> None:
        # Chroma metadata 必须扁平；text 放 documents 供 where/document 检索。
        flat_meta = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        flat_meta[EXPIRES_FIELD] = float(expires_at)
        self._col.upsert(
            ids=[event_id],
            embeddings=[vector],
            documents=[text or event_id],
            metadatas=[flat_meta],
        )

    def search(
        self,
        vector: List[float],
        *,
        limit: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        res = self._col.query(
            query_embeddings=[vector],
            n_results=limit,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        out: List[Dict[str, Any]] = []
        ids = res.get("ids") or [[]]
        metas = res.get("metadatas") or [[]]
        docs = res.get("documents") or [[]]
        dists = res.get("distances") or [[]]
        for i, event_id in enumerate(ids[0]):
            meta = metas[0][i] or {}
            out.append(
                {
                    "event_id": event_id,
                    "score": float(1.0 - (dists[0][i] if len(dists[0]) > i else 0.0)),
                    "text": docs[0][i] if len(docs[0]) > i else "",
                    "metadata": meta,
                }
            )
        return out

    def touch(self, event_id: str, expires_at: float) -> None:
        # 续命 = 只更新到期元数据（deadline+duration 已被上层算成 expires_at）
        self._col.update(ids=[event_id], metadatas=[{EXPIRES_FIELD: float(expires_at)}])

    def delete_expired(self, now: Optional[float] = None) -> int:
        """把“删除”交给 DB：一条 where 删除，不逐个 pop。"""
        now = time.time() if now is None else float(now)
        before = self._col.count()
        self._col.delete(where={EXPIRES_FIELD: {"$lt": now}})
        after = self._col.count()
        return before - after

    def delete(self, event_id: str) -> None:
        self._col.delete(ids=[event_id])

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        # PersistentClient 会自动落盘；预留关闭钩子。
        return None


def build_default_vector_db(
    config: VectorDbConfig,
    backend: str = "chroma",
) -> VectorDb:
    """按名构造后端。后续要换 Milvus/Qdrant/LanceDB 就扩展这里，调用方不变。"""
    if backend == "chroma":
        persist = str(Path(config.persist_dir).expanduser())
        return ChromaVectorDb(
            collection=config.collection,
            persist_dir=persist,
            distance=config.distance,
        )
    raise ValueError(f"unsupported vector db backend: {backend}")
