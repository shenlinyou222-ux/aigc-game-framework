# -*- coding: utf-8 -*-
"""svc_status.py — 本机 agent 服务健康聚合探针（一条命令看清 7 个端口）。

用法:
    python svc_status.py            # JSON 输出
    python svc_status.py --table    # 人类可读表格

探针设计: HTTP 服务打各自 /health(或等价)端点; 纯 socket 服务(Blender addon 9876)
只做 TCP connect。整体 3s 超时, 任何单个服务挂掉不影响其它探针。
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request

SERVICES = [
    # name,        kind,   target,                                 probe
    ("ctx8000",    "http", "http://127.0.0.1:8000/ctx/status",      "ctx 记忆/召回后端"),
    ("ctx-router", "http", "http://127.0.0.1:8001/health",          "小模型意图路由"),
    ("voice-stt",  "http", "http://127.0.0.1:8019/health",          "统一语音转文字入口"),
    ("stt-asr",    "http", "http://127.0.0.1:8124/health",          "Qwen3-ASR llama.cpp"),
    ("ollama",     "http", "http://127.0.0.1:11434/api/version",    "embedding/小模型"),
    ("dsh-web",    "tcp",  ("127.0.0.1", 3080),                     "DSH web UI"),
    ("blender-mcp", "tcp", ("127.0.0.1", 9876),                     "Blender addon socket"),
]

TIMEOUT = 3.0


def _probe_http(url: str) -> dict:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            body = r.read(4096)
        return {"alive": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "detail": body.decode("utf-8", "replace")[:200]}
    except Exception as exc:  # noqa: BLE001 - 探针吃掉一切异常
        return {"alive": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "detail": repr(exc)}


def _probe_tcp(host: str, port: int) -> dict:
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            return {"alive": True,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "detail": f"tcp {host}:{port} connect ok"}
    except Exception as exc:  # noqa: BLE001
        return {"alive": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "detail": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent services health aggregate")
    parser.add_argument("--table", action="store_true", help="人类可读表格输出")
    args = parser.parse_args()

    results = {}
    for name, kind, target, note in SERVICES:
        if kind == "http":
            probe = _probe_http(target)
        else:
            probe = _probe_tcp(*target)
        results[name] = {"note": note, "kind": kind, **probe}

    # blender-mcp 为可选: Blender 未启动时本就关闭, 不计入整体 ok
    payload = {"ok": all(r["alive"] for k, r in results.items() if k != "blender-mcp"),
               "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "services": results}
    if args.table:
        print(f"{'service':<14}{'alive':<7}{'ms':>8}  note")
        for name, r in results.items():
            opt = " (optional)" if name == 'blender-mcp' else ""
            print(f"{name:<14}{str(r['alive']):<7}{r['latency_ms']:>8}  {r['note']}{opt}")
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
