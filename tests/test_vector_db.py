"""向量数据库适配层测试。

重点验证分层契约：真正向量库装好后，到期清理是**数据库 delete**，不是应用层遍历 pop。
本环境未安装 chromadb，因此只验证：
- 未装库时构造 ChromaVectorDb 给出清晰报错（而不是悄悄走 JSONL 假库）；
- 后端工厂对未知后端报错；
- 事件/元数据字段名契约存在（expires_at 进元数据）。

若本机装了 chromadb，可加一个集成用例连本地 PersistentClient。
"""

from __future__ import annotations

import unittest

from aigc_core.vector_db import (
    EXPIRES_FIELD,
    KIND_FIELD,
    SESSION_FIELD,
    ChromaVectorDb,
    VectorDbConfig,
    build_default_vector_db,
)


class VectorDbContractTests(unittest.TestCase):
    def test_reserved_field_names_present(self) -> None:
        # 生命周期到期时间必须落在记录元数据里，DB 靠它删。
        self.assertEqual(EXPIRES_FIELD, "expires_at")
        self.assertEqual(SESSION_FIELD, "session_id")
        self.assertEqual(KIND_FIELD, "kind")

    def test_unknown_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_default_vector_db(VectorDbConfig(), backend="not-a-db")

    def test_chroma_unavailable_gives_clear_error(self) -> None:
        # 未安装 chromadb 时应明确报“装库”，而不是默默退回 JSONL 假库。
        with self.assertRaises(RuntimeError) as ctx:
            ChromaVectorDb()
        self.assertIn("chromadb is not installed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
