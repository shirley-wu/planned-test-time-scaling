"""Start/stop a local `vllm serve` process and hand out OpenAI-compatible clients for it."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from openai import OpenAI

# Passed to `vllm serve --override-generation-config` so the server's defaults do not depend on the
# model's generation_config.json. Requests set temperature / max_tokens explicitly anyway.
DEFAULT_GENERATION_CONFIG = {
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "max_new_tokens": 20000,
    "do_sample": True,
}


class VLLMClient:
    def __init__(
        self,
        model_path: str,
        served_model_name: str | None = None,
        port: int = 8000,
        host: str = "127.0.0.1",
        api_key: str = "EMPTY",
        request_timeout_s: int = 3600,
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name or model_path
        self.port = port
        self.host = host
        self.api_key = api_key
        self.request_timeout_s = request_timeout_s
        self.base_url = f"http://{host}:{port}/v1"

        self._thread_local = threading.local()
        self._server_proc: subprocess.Popen | None = None
        self._server_log_file = None

    def openai_client(self) -> OpenAI:
        """One OpenAI client per thread (safe to call from a ThreadPoolExecutor)."""
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.request_timeout_s)
            self._thread_local.client = client
        return client

    def start_server(
        self,
        *,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float | None = None,
        cuda_visible_devices: str | None = None,
        log_path: str | Path | None = None,
        startup_timeout_s: int = 600,
        poll_interval_s: float = 2.0,
    ) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            raise RuntimeError("vLLM server is already running in this client.")

        cmd = [
            "vllm", "serve", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--data-parallel-size", str(data_parallel_size),
            "--served-model-name", self.served_model_name,
            "--override-generation-config", json.dumps(DEFAULT_GENERATION_CONFIG),
        ]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]

        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._server_log_file = open(log_path, "w")
            stdout = stderr = self._server_log_file
        else:
            stdout = stderr = subprocess.DEVNULL

        print("starting vLLM:", " ".join(cmd))
        self._server_proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True, text=True)
        self.wait_until_ready(timeout_s=startup_timeout_s, poll_interval_s=poll_interval_s)

    def wait_until_ready(self, *, timeout_s: int = 600, poll_interval_s: float = 2.0) -> None:
        deadline = time.time() + timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            if self._server_proc is not None and self._server_proc.poll() is not None:
                raise RuntimeError("vLLM server exited before becoming ready (see its log).")
            try:
                available = [m.id for m in self.openai_client().models.list().data]
                if self.served_model_name in available:
                    return
                last_error = RuntimeError(f"model {self.served_model_name!r} not in {available}")
            except Exception as exc:  # server not up yet
                last_error = exc
            time.sleep(poll_interval_s)
        raise TimeoutError(f"vLLM server not ready after {timeout_s}s. Last error: {last_error}")

    def stop_server(self, *, graceful_timeout_s: int = 20) -> None:
        if self._server_proc is None:
            return
        if self._server_proc.poll() is None:
            os.killpg(self._server_proc.pid, signal.SIGTERM)
            try:
                self._server_proc.wait(timeout=graceful_timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self._server_proc.pid, signal.SIGKILL)
                self._server_proc.wait()
        self._server_proc = None
        if self._server_log_file is not None:
            self._server_log_file.close()
            self._server_log_file = None
