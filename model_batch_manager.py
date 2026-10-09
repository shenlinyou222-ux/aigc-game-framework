# -*- coding: utf-8 -*-
"""动态批处理与模型常驻管理(2026-09-07)。

8GB 单卡纪律下的 Ollama 资源调度,keep_alive/批处理策略的唯一入口:
- qwen3.5:0.8b 非常驻: 每请求携带 keep_alive=空闲窗(默认 45s),批后无新请求即自动卸载;
  环境变量 MBM_STRICT=1 切 0 秒即用即卸。取代旧 llm_json keep_alive=-1 的 Forever 常驻
  (实测 2026-09-07 ollama ps 显示 UNTIL=Forever,训练前必须手动腾卡,本模块接管后自动化)。
- qwen3-embedding:0.6b 实时加载: 常驻位(默认 30m),被挤出后下次调用现场重载(冷载耗时计入
  统计);真批处理——多条输入去重合并为一次 /api/embed,超过 max_batch 自动分片。
- 训练闸门: pretrain() 封闸→等在途请求清零→卸载全部已加载模型→校验 /api/ps 清空,
  特化训练独占显卡;posttrain() 预热常驻位并放行。

HTTP 仅标准库 urllib(与 hot_dialog 同款 UTF-8 直连,规避 Git Bash curl 编码坑);
所有函数 base_url 可注入,单测用桩服务器。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 策略(环境变量可覆盖)
# ---------------------------------------------------------------------------

QWEN_MODEL = "qwen3.5:0.8b"
EMBED_MODEL = "qwen3-embedding:0.6b"

# 非常驻:最后一条请求起算,窗口内无新请求 Ollama 自动卸载;strict 模式直接 0(即用即卸)
QWEN_KEEP_ALIVE = os.environ.get("MBM_QWEN_KEEP_ALIVE", "45s")
# 实时加载常驻位:语音情绪路由在链路上,冷载 5~8s 不可接受,故常驻;被训练/大模型挤出后现场重载
EMBED_KEEP_ALIVE = os.environ.get("MBM_EMBED_KEEP_ALIVE", "30m")
STRICT_UNLOAD = os.environ.get("MBM_STRICT", "") == "1"

FLUSH_WINDOW = float(os.environ.get("MBM_FLUSH_WINDOW", "0.12"))  # 单条 embed 攒批窗口(秒)
EMBED_MAX_BATCH = int(os.environ.get("MBM_EMBED_MAX_BATCH", "32"))  # 单次 /api/embed 最大条数

DEFAULT_BASE = "http://127.0.0.1:11434"

STATS: Dict[str, int] = {"chat_cold": 0, "chat_warm": 0, "embed_cold": 0, "embed_warm": 0,
                         "embed_dup_skipped": 0, "embed_batches": 0}


class TrainGateClosed(RuntimeError):
    """训练闸门已关闭(pretrain 独占显卡期间拒绝新推理请求)。"""


# ---------------------------------------------------------------------------
# Ollama HTTP 原语
# ---------------------------------------------------------------------------

def _post(base_url: str, path: str, body: Dict[str, Any], timeout: float = 180) -> Dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(base_url: str, path: str, timeout: float = 10) -> Dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read())


def ps(base_url: str = DEFAULT_BASE) -> Dict[str, Dict[str, Any]]:
    """当前已加载模型:name -> {size, size_vram, expires_at, ...}。"""
    data = _get(base_url, "/api/ps")
    return {m.get("name", m.get("model", "?")): m for m in data.get("models", [])}


def is_loaded(name: str, base_url: str = DEFAULT_BASE) -> bool:
    return name in ps(base_url)


def policy_keep_alive(model: str = QWEN_MODEL, strict: Optional[bool] = None) -> "str | int":
    """keep_alive 策略解析(llm_json/embedders/hot_dialog 统一从这里取)。

    注意:Ollama 服务端是 Go duration 解析,整数 -1/0 必须传 int,字符串会被 400。
    """
    s = STRICT_UNLOAD if strict is None else strict
    if s:
        return 0
    if model == QWEN_MODEL:
        return QWEN_KEEP_ALIVE
    if model == EMBED_MODEL:
        return EMBED_KEEP_ALIVE
    return QWEN_KEEP_ALIVE


# ---------------------------------------------------------------------------
# 常驻控制:unload / load / 训练闸门
# ---------------------------------------------------------------------------

def unload(name: str, base_url: str = DEFAULT_BASE) -> None:
    """立即卸载一个模型(keep_alive=0 的空 chat 请求是官方卸载姿势)。"""
    _post(base_url, "/api/chat", {"model": name, "keep_alive": 0}, timeout=30)


def unload_all(base_url: str = DEFAULT_BASE) -> List[str]:
    """卸载当前全部已加载模型,返回卸载清单。"""
    names = list(ps(base_url).keys())
    for n in names:
        unload(n, base_url)
    return names


def load(name: str, base_url: str = DEFAULT_BASE, keep_alive: Optional[Any] = None) -> float:
    """现场加载模型,返回加载耗时(秒)。embedding 模型用空 input 加载。"""
    ka = keep_alive if keep_alive is not None else policy_keep_alive(name)
    t0 = time.time()
    if name == EMBED_MODEL:
        _post(base_url, "/api/embed", {"model": name, "input": "预热", "keep_alive": ka}, timeout=300)
    else:
        _post(base_url, "/api/chat", {"model": name, "messages": [{"role": "user", "content": "ping"}],
                                      "stream": False, "think": False, "keep_alive": ka,
                                      "options": {"num_predict": 1}}, timeout=300)
    return time.time() - t0


_GATE = threading.Event()
_GATE.set()
_INFLIGHT = 0
_INFLIGHT_CV = threading.Condition()


def _gate_check() -> None:
    if not _GATE.is_set():
        raise TrainGateClosed("pretrain 独占显卡中;跑 model_batch_manager.py posttrain 恢复")


def pretrain(base_url: str = DEFAULT_BASE, verify: bool = True) -> List[str]:
    """训练前腾卡:封闸 → 等在途请求清零 → 卸载全部 → 校验。返回卸载清单。"""
    global _INFLIGHT
    _GATE.clear()
    with _INFLIGHT_CV:
        _INFLIGHT_CV.wait_for(lambda: _INFLIGHT == 0, timeout=60)
    unloaded = unload_all(base_url)
    if verify:
        deadline = time.time() + 20          # 服务端卸载是异步的(ps 先显示 Stopping...), 轮询等清空
        left = ps(base_url)
        while left and time.time() < deadline:
            time.sleep(0.5)
            left = ps(base_url)
        if left:
            _GATE.set()
            raise RuntimeError(f"卸载后仍有残留: {list(left.keys())}")
    return unloaded


def posttrain(base_url: str = DEFAULT_BASE, preload_embed: bool = True) -> None:
    """训练后恢复:预热 embedding 常驻位并放行闸门。"""
    if preload_embed and not is_loaded(EMBED_MODEL, base_url):
        load(EMBED_MODEL, base_url)
    _GATE.set()


# ---------------------------------------------------------------------------
# 动态批处理:chat(串行队列语义 + 冷热统计) / embed(去重+攒批+分片)
# ---------------------------------------------------------------------------

def chat(messages: List[Dict[str, str]], *, model: str = QWEN_MODEL, system: Optional[str] = None,
         options: Optional[Dict[str, Any]] = None, think: bool = False,
         fmt: Any = None, keep_alive: Any = None, base_url: str = DEFAULT_BASE,
         timeout: float = 180) -> Dict[str, Any]:
    """单次 chat;keep_alive 缺省走策略(非常驻)。并发调用由 Ollama 排队,一次加载服务整批。"""
    _gate_check()
    if system and (not messages or messages[0].get("role") != "system"):
        messages = [{"role": "system", "content": system}] + list(messages)
    body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                            "think": think, "keep_alive": policy_keep_alive(model) if keep_alive is None else keep_alive}
    if options:
        body["options"] = options
    if fmt is not None:
        body["format"] = fmt
    if not is_loaded(model, base_url):
        STATS["chat_cold"] += 1
    else:
        STATS["chat_warm"] += 1
    with _INFLIGHT_CV:
        globals()["_INFLIGHT"] = _INFLIGHT + 1
    try:
        return _post(base_url, "/api/chat", body, timeout=timeout)
    finally:
        with _INFLIGHT_CV:
            globals()["_INFLIGHT"] = _INFLIGHT - 1
            _INFLIGHT_CV.notify_all()


def chat_text(messages: List[Dict[str, str]], **kw) -> str:
    """chat 的取词便捷版(与 hot_dialog.chat 同款后处理)。"""
    resp = chat(messages, **kw)
    return str(resp.get("message", {}).get("content", "")).strip()


def chat_many(specs: List[Dict[str, Any]], **kw) -> List[str]:
    """批量逐条生成(逐句台词场景):一次冷载服务整批,内部串行走 Ollama 单模型队列。"""
    return [chat_text(s["messages"], **{**kw, **{k: v for k, v in s.items() if k != "messages"}})
            for s in specs]


def embed(texts: List[str], *, model: str = EMBED_MODEL, keep_alive: Any = None,
          base_url: str = DEFAULT_BASE, max_batch: int = EMBED_MAX_BATCH,
          timeout: float = 180) -> List[List[float]]:
    """批量嵌入:文本去重(重复 query 在日志/检索里高频出现)→ 分片 → 一次请求一批。

    返回顺序与入参对齐。任一分片失败整体抛出,由调用方决定降级。
    """
    _gate_check()
    uniq = list(dict.fromkeys(texts))
    STATS["embed_dup_skipped"] += len(texts) - len(uniq)
    ka = policy_keep_alive(model) if keep_alive is None else keep_alive
    if not is_loaded(model, base_url):
        STATS["embed_cold"] += 1
    else:
        STATS["embed_warm"] += 1
    vecs: Dict[str, List[float]] = {}
    for i in range(0, len(uniq), max_batch):
        chunk = uniq[i:i + max_batch]
        STATS["embed_batches"] += 1
        data = _post(base_url, "/api/embed", {"model": model, "input": chunk, "keep_alive": ka},
                     timeout=timeout)
        out = data.get("embeddings") or data.get("data")
        if not out or len(out) != len(chunk):
            got = len(out) if out else 0
            raise RuntimeError(f"ollama batch embed mismatch: got {got}, want {len(chunk)}")
        vecs.update({t: [float(x) for x in v] for t, v in zip(chunk, out)})
    return [vecs[t] for t in texts]


class EmbedMicroBatcher:
    """单条 embed 攒批器:FLUSH_WINDOW 内到达的零散单条合并为一次 /api/embed。

    服务多线程(如 voice_stt 并发转写→情绪路由)时收益最大;单线程调用方直接用 embed()。
    """

    def __init__(self, flush: float = FLUSH_WINDOW, max_batch: int = EMBED_MAX_BATCH,
                 base_url: str = DEFAULT_BASE):
        self.flush, self.max_batch, self.base_url = flush, max_batch, base_url
        self._buf: List[Any] = []  # [(text, holder_dict)]
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None

    def _ensure_thread(self) -> None:
        with self._cv:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True,
                                                name="mbm-embed-batcher")
                self._thread.start()

    def submit(self, text: str, timeout: float = 120) -> List[float]:
        _gate_check()
        holder: Dict[str, Any] = {}
        with self._cv:
            self._buf.append((text, holder))
            self._cv.notify()
        self._ensure_thread()
        deadline = time.time() + timeout
        while "vec" not in holder and "err" not in holder:
            if time.time() > deadline:
                raise TimeoutError("embed_one 超时")
            time.sleep(0.005)
        if "err" in holder:
            raise holder["err"]
        return holder["vec"]

    def _loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait(timeout=self.flush)
                if not self._buf:
                    continue
                time.sleep(self.flush)  # 再攒一窗,合并后到的单条
                batch, self._buf = self._buf[:self.max_batch], self._buf[self.max_batch:]
            texts = [t for t, _ in batch]
            try:
                vecs = embed(texts, base_url=self.base_url, max_batch=self.max_batch)
                for (_, h), v in zip(batch, vecs):
                    h["vec"] = v
            except Exception as exc:  # 整批同错(与 embed_texts 语义一致)
                for _, h in batch:
                    h["err"] = exc


_micro: Optional[EmbedMicroBatcher] = None
_micro_lock = threading.Lock()


def embed_one(text: str, **kw) -> List[float]:
    """单条嵌入走攒批器(模块级单例);批内去重与分片复用 embed()。"""
    global _micro
    if _micro is None:
        with _micro_lock:
            if _micro is None:
                _micro = EmbedMicroBatcher(base_url=kw.pop("base_url", DEFAULT_BASE))
    return _micro.submit(text, timeout=kw.pop("timeout", 120))


# ---------------------------------------------------------------------------
# CLI:status / unload / load / pretrain / posttrain / bench
# ---------------------------------------------------------------------------

def status(base_url: str = DEFAULT_BASE) -> str:
    lines = [f"[策略] qwen3.5 非常驻 keep_alive={policy_keep_alive(QWEN_MODEL)!r}"
             f"(strict={STRICT_UNLOAD}) | embedding 常驻位 keep_alive={policy_keep_alive(EMBED_MODEL)!r}",
             f"[计数] {STATS}"]
    resident = ps(base_url)
    if resident:
        for n, m in resident.items():
            lines.append(f"[驻留] {n}  VRAM={m.get('size_vram', '?')}  UNTIL={m.get('expires_at', '?')}")
    else:
        lines.append("[驻留] (空) — 显卡已腾净")
    return "\n".join(lines)


def bench(base_url: str = DEFAULT_BASE) -> str:
    """冷/热延迟对比:chat 单条 + embed 32 条批。会临时加载模型,结束恢复非常驻。"""
    out = []
    unload(QWEN_MODEL, base_url) if is_loaded(QWEN_MODEL, base_url) else None
    t0 = time.time()
    chat_text([{"role": "user", "content": "只回复:好"}], options={"num_predict": 4}, base_url=base_url)
    out.append(f"chat 冷载+生成: {time.time()-t0:.2f}s")
    t0 = time.time()
    chat_text([{"role": "user", "content": "只回复:好"}], options={"num_predict": 4}, base_url=base_url)
    out.append(f"chat 热请求:     {time.time()-t0:.2f}s")
    texts = [f"第{i}条测试文本" for i in range(32)]
    t0 = time.time()
    embed(texts, base_url=base_url)
    out.append(f"embed 冷载+32条批: {time.time()-t0:.2f}s")
    t0 = time.time()
    embed([texts[0]] * 8 + ["新文本"], base_url=base_url)
    out.append(f"embed 热请求+去重(9入2算): {time.time()-t0:.2f}s")
    out.append(f"[计数] {STATS}")
    return "\n".join(out)


def main() -> None:
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "status":
        print(status())
    elif cmd == "unload":
        names = [arg] if arg else None
        print("已卸载:", names or unload_all() or "(无驻留)")
    elif cmd == "load":
        if not arg:
            raise SystemExit("用法: load <model>")
        print(f"{arg} 加载耗时 {load(arg):.2f}s")
    elif cmd == "pretrain":
        print("已卸载:", pretrain() or "(无驻留)", "→ 显卡已腾净,可开训练")
    elif cmd == "posttrain":
        posttrain()
        print("已恢复:", status())
    elif cmd == "bench":
        print(bench())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
