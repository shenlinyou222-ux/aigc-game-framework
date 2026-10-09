# -*- coding: utf-8 -*-
"""语音转文字微服务: 复用 voice_intent.py 的 transcribe(唯一后端 Qwen3-ASR-0.6B)
用法: python voice_stt_server.py        (默认 127.0.0.1:8019)
接口:
  GET  /health  -> {"ok": true}
  POST /stt     -> body: PCM16LE 单声道 16kHz 裸流 -> {"text": "..."}
CORS 全放行(供 DSH web 界面 127.0.0.1:3080 跨域调用)。
模型在首个请求时惰性加载(与 voice_intent.py 相同后端, 各持一份进程内实例)。
"""
import io
import json
import os
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_intent import transcribe, ASR_BACKEND, STT_URL   # noqa: E402

HOST, PORT = "127.0.0.1", 8019
_tlock = threading.Lock()          # funasr/whisper 模型非线程安全, 串行转写


def _pcm_to_wav(pcm: bytes, rate: int = 16000) -> str:
    """裸 PCM16LE 单声道 -> 临时 wav 文件(voice_intent.transcribe 吃文件路径)"""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="dsh_stt_")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            backend_ok = False
            try:
                with urllib.request.urlopen(STT_URL + "/health", timeout=1.5) as r:
                    backend_ok = b"ok" in r.read()
            except Exception:
                backend_ok = False
            body = json.dumps({"ok": backend_ok, "backend": ASR_BACKEND,
                               "backend_alive": backend_ok}).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/stt":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            rate = 16000
            try:
                if "rate" in q:
                    rate = int(min(96000, max(8000, int(q["rate"][0]))))
            except (ValueError, IndexError):
                rate = 16000
            pcm = self.rfile.read(n)
            if len(pcm) < 3200:            # <0.1s 的空段直接回空
                raise ValueError("audio too short")
            wav = _pcm_to_wav(pcm, rate)
            try:
                with _tlock:
                    text = transcribe(wav)
            finally:
                try:
                    os.remove(wav)
                except OSError:
                    pass
            body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as e:             # noqa: BLE001
            body = json.dumps({"error": repr(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):     # 安静模式: 只打异常, 不刷屏
        pass


if __name__ == "__main__":
    print(f"[voice-stt] listening http://{HOST}:{PORT}  backend={ASR_BACKEND}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
