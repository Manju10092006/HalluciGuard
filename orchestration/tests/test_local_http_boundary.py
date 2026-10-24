"""Actual loopback HTTP/auth/storage boundary; no graph, ML or provider calls."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]


def request(base, path, method="GET", body=None, token=None, headers=None):
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=data, method=method, headers=request_headers)
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        raw = response.read()
        content = json.loads(raw) if raw and "application/json" in response.headers.get("Content-Type", "") else raw.decode()
        return response.status, content, response.headers


@contextmanager
def local_api(db_path, jwt_secret):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {
        **os.environ,
        "AUTH_DB_PATH": str(db_path),
        "JWT_SECRET": jwt_secret,
        "PYTHON_DOTENV_DISABLED": "1",
        "HALLUCIGUARD_FREE_ONLY": "true",
        "HALLUCIGUARD_LLM_PROVIDER_ORDER": "openrouter",
        "HALLUCIGUARD_LLM_MODEL": "openrouter/free",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
        "OPENROUTER_API_KEY": "",
        "GOOGLE_CLIENT_ID": "",
        "GOOGLE_CLIENT_SECRET": "",
        "CORS_ORIGINS": "http://localhost:3000",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    # Bind loopback only and keep subprocess output private: credentials and
    # authenticated request bodies must never be included in test diagnostics.
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "orchestration.api:app", "--host",
         "127.0.0.1", "--port", str(port), "--no-access-log", "--log-level", "error"],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"Local API exited during startup: {process.returncode}")
            try:
                if request(base, "/")[0] == 200:
                    break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.05)
        else:
            raise AssertionError("Local API did not become reachable within 20 seconds")
        yield base
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_real_http_auth_isolation_validation_and_restart_persistence(tmp_path):
    db_path = tmp_path / "auth.db"
    jwt_secret = secrets.token_urlsafe(48)
    alice = {"email": "local-alice@example.test", "password": "local-test-password", "name": "Alice"}
    bob = {"email": "local-bob@example.test", "password": "local-test-password", "name": "Bob"}
    with local_api(db_path, jwt_secret) as base:
        assert request(base, "/auth/me")[0] == 401
        assert request(base, "/api/history", token="invalid.jwt.value")[0] == 401
        assert request(base, "/verify", "POST", {"user_query": ""})[0] == 422
        assert request(base, "/verify", "POST", {"user_query": "test", "generation_mode": "invalid"})[0] == 422
        status, registered, _ = request(base, "/auth/register", "POST", alice)
        assert status == 200
        alice_token = registered["access_token"]
        status, other, _ = request(base, "/auth/register", "POST", bob)
        assert status == 200
        bob_token = other["access_token"]
        assert request(base, "/auth/login", "POST", {**alice, "password": "wrong"})[0] == 401
        status, me, _ = request(base, "/auth/me", token=alice_token)
        assert status == 200 and me["user"]["id"] == registered["user"]["id"]
        # This is conversation history, not verified Memory: explicitly mark the
        # synthetic record unverified and assert ownership through real HTTP.
        record = {"query": "local history boundary", "result": {"verification_status": "unverified"}}
        assert request(base, "/api/history", "POST", record, alice_token)[0] == 200
        assert len(request(base, "/api/history", token=alice_token)[1]["history"]) == 1
        assert request(base, "/api/history", token=bob_token)[1]["history"] == []
        status, _, headers = request(base, "/auth/me", "OPTIONS", headers={
            "Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization"})
        assert status == 200 and headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert request(base, "/auth/me", "OPTIONS", headers={
            "Origin": "https://untrusted.example", "Access-Control-Request-Method": "GET"})[0] == 400
    assert db_path.is_file()
    with local_api(db_path, jwt_secret) as base:
        status, login, _ = request(base, "/auth/login", "POST", alice)
        assert status == 200
        history = request(base, "/api/history", token=login["access_token"])[1]["history"]
        assert len(history) == 1 and history[0]["verdict"] == "unverified"
        assert request(base, "/api/history", token=bob_token)[1]["history"] == []
