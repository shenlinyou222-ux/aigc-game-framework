# -*- coding: utf-8 -*-
"""model_batch_manager 单测:桩 Ollama 服务器验证批处理合并、去重、keep_alive 策略与训练闸门。

运行: cd aigc_demo_framework && E:/py312/python.exe -m unittest tests.test_batch_manager -v
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import model_batch_manager as mbm


class _StubOllama:
    """最小 Ollama 桩:记录请求、模拟模型驻留状态(keep_alive=0 即卸载)。"""

    def __init__(self):
        self.requests = []          # [(path, body)]
        self.loaded = set()
        self.lock = threading.Lock()

    def handle(self, method, path, body):
        with self.lock:
            self.requests.append((path, body))
            if path == "/api/ps" and method == "GET":
                return {"models": [{"name": n, "size_vram": "1MB"} for n in sorted(self.loaded)]}
            if method != "POST":
                return {}
            model = body.get("model", "")
            if body.get("keep_alive") == 0:
                self.loaded.discard(model)
                return {"message": {"content": ""}}
            if path == "/api/chat":
                self.loaded.add(model)
                return {"message": {"content": "测试台词"}}
            if path == "/api/generate":
                self.loaded.add(model)
                return {"response": '{"ok": 1}'}
            if path == "/api/embed":
                self.loaded.add(model)
                n = len(body["input"]) if isinstance(body["input"], list) else 1
                return {"embeddings": [[0.1, 0.2]] * n}
        return {}

    def calls(self, path):
        with self.lock:
            return [(p, b) for p, b in self.requests if p == path]


class _Handler(BaseHTTPRequestHandler):
    stub: _StubOllama = None  # 注入

    def log_message(self, *a):
        pass

    def _reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(self.stub.handle("GET", self.path.split("?")[0], {}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        self._reply(self.stub.handle("POST", self.path.split("?")[0], body))


class TestBatchManager(unittest.TestCase):
    def setUp(self):
        self.stub = _StubOllama()
        _Handler.stub = self.stub
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        for k in mbm.STATS:
            mbm.STATS[k] = 0
        mbm.STRICT_UNLOAD = False
        mbm.QWEN_KEEP_ALIVE = "45s"
        mbm.EMBED_KEEP_ALIVE = "30m"

    def tearDown(self):
        self.server.shutdown()

    # ---- 策略解析 ----
    def test_policy_default_nonresident(self):
        self.assertEqual(mbm.policy_keep_alive(mbm.QWEN_MODEL), "45s")
        self.assertEqual(mbm.policy_keep_alive(mbm.EMBED_MODEL), "30m")

    def test_policy_strict_zero(self):
        self.assertEqual(mbm.policy_keep_alive(mbm.QWEN_MODEL, strict=True), 0)
        self.assertEqual(mbm.policy_keep_alive(mbm.EMBED_MODEL, strict=True), 0)

    # ---- chat:keep_alive 策略下发 + 冷热统计 ----
    def test_chat_attaches_policy(self):
        self.assertFalse(mbm.is_loaded(mbm.QWEN_MODEL, self.base))  # 调用前未驻留→冷载
        txt = mbm.chat_text([{"role": "user", "content": "hi"}], base_url=self.base)
        self.assertEqual(txt, "测试台词")
        body = self.stub.calls("/api/chat")[-1][1]
        self.assertEqual(body["keep_alive"], "45s")
        self.assertEqual(body["think"], False)
        self.assertEqual(mbm.STATS["chat_cold"], 1)
        mbm.chat_text([{"role": "user", "content": "hi"}], base_url=self.base)
        self.assertEqual(mbm.STATS["chat_warm"], 1)                # 第二次走驻留热请求

    def test_chat_many_serves_batch(self):
        out = mbm.chat_many([{"messages": [{"role": "user", "content": f"m{i}"}]} for i in range(3)],
                            base_url=self.base)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(self.stub.calls("/api/chat")), 3)

    # ---- embed:去重 + 分片 + 顺序对齐 ----
    def test_embed_dedup_and_order(self):
        vecs = mbm.embed(["a", "b", "a", "c"], base_url=self.base)
        self.assertEqual(len(vecs), 4)
        self.assertEqual(vecs[0], vecs[2])
        self.assertEqual(vecs[1], [0.1, 0.2])
        sent = self.stub.calls("/api/embed")[-1][1]["input"]
        self.assertEqual(sent, ["a", "b", "c"])                    # 去重后只发 3 条
        self.assertEqual(mbm.STATS["embed_dup_skipped"], 1)

    def test_embed_chunking(self):
        texts = [f"t{i}" for i in range(70)]
        mbm.embed(texts, base_url=self.base, max_batch=32)
        sizes = [len(b[1]["input"]) for b in self.stub.calls("/api/embed")]
        self.assertEqual(sizes, [32, 32, 6])

    # ---- 单条攒批:并发单条合并一次请求 ----
    def test_embed_one_coalesces(self):
        batcher = mbm.EmbedMicroBatcher(flush=0.15, base_url=self.base)
        results = [[]] * 2

        def worker(i):
            results[i] = batcher.submit(f"s{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        threads[0].start()
        threading.Event().wait(0.03)                               # 第二条在攒批窗内到达
        threads[1].start()
        for t in threads:
            t.join(timeout=10)
        embed_calls = self.stub.calls("/api/embed")
        self.assertEqual(len(embed_calls), 1)                      # 合并为一次请求
        self.assertEqual(sorted(embed_calls[0][1]["input"]), ["s0", "s1"])
        self.assertEqual(results[0], [0.1, 0.2])

    # ---- 常驻控制:unload_all / 训练闸门 ----
    def test_unload_all_and_gate(self):
        mbm.chat_text([{"role": "user", "content": "hi"}], base_url=self.base)      # qwen 进驻留
        mbm.embed(["x"], base_url=self.base)                                        # embedding 进驻留
        self.assertEqual(len(self.stub.loaded), 2)
        unloaded = mbm.pretrain(base_url=self.base)
        self.assertEqual(len(unloaded), 2)
        self.assertEqual(len(self.stub.loaded), 0)                 # 显卡腾净
        unload_bodies = [b for _, b in self.stub.calls("/api/chat") if b.get("keep_alive") == 0]
        self.assertEqual(len(unload_bodies), 2)
        with self.assertRaises(mbm.TrainGateClosed):               # 封闸期间拒绝新推理
            mbm.chat_text([{"role": "user", "content": "hi"}], base_url=self.base)
        mbm.posttrain(base_url=self.base)                          # 预热 embedding 常驻位+放行
        self.assertIn(mbm.EMBED_MODEL, self.stub.loaded)
        self.assertEqual(mbm.chat_text([{"role": "user", "content": "hi"}], base_url=self.base),
                         "测试台词")


if __name__ == "__main__":
    unittest.main()
