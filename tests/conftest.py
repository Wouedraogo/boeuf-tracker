"""
conftest.py — Pytest fixtures for Boeuf Tracker integration tests.

Fixtures:
  * app / client            — Flask test client (TESTING=True)
  * tmp_state               — isolated AppState copy for one test
  * tmp_db_path             — temp pickle path, auto-cleanup
  * sample_embedding        — deterministic 16-dim float32 vector
  * sample_video            — short synthetic .mp4 for capture tests
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Flask app ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def app():
    """Lazy-import app to avoid loading YOLO/MLX at test time.

    We monkeypatch the heavy detection-thread launcher so importing app.py
    does not require GPU / MLX / network. The Flask routes themselves are
    exercised via the test client.
    """
    # Block the real detection thread + analytics init from launching
    import core.processor as proc_mod

    def _noop_start(*a, **kw):
        return None

    proc_mod.start_detection_thread = _noop_start

    import app as app_mod
    app_mod.app.config.update(TESTING=True)
    yield app_mod.app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


# ─── Temp state isolation ──────────────────────────────────────────────────
@pytest.fixture
def tmp_state(monkeypatch):
    """Replace global STATE with a fresh AppState for the duration of one test."""
    from utils.state import AppState
    fresh = AppState()
    import utils.state as state_mod
    monkeypatch.setattr(state_mod, "STATE", fresh)
    return fresh


# ─── Temp DB path ──────────────────────────────────────────────────────────
@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a path inside pytest's tmp_path, never created."""
    return tmp_path / "test_cattle_db.pkl"


# ─── Sample data ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_embedding():
    """Deterministic 16-dim L2-normalized embedding."""
    rng = np.random.default_rng(seed=42)
    v = rng.random(16).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


@pytest.fixture
def sample_embedding_pair():
    """Two embeddings: one close, one far from sample_embedding."""
    rng = np.random.default_rng(seed=43)
    base = rng.random(16).astype(np.float32)
    base /= np.linalg.norm(base) + 1e-8
    close = base * 0.95 + 0.05 * rng.random(16).astype(np.float32)
    close /= np.linalg.norm(close) + 1e-8
    far = -base
    return base, close, far


@pytest.fixture
def sample_video(tmp_path):
    """Create a tiny synthetic .mp4 for capture tests."""
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python not installed")
    out = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h, fps = 64, 48, 10
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("VideoWriter unavailable on this platform")
    try:
        for i in range(15):  # 1.5s of frames
            frame = np.full((h, w, 3), (i * 17) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return out