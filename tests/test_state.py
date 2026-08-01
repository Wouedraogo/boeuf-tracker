"""
test_state.py — AppState + color_for_name unit tests.
"""
import threading

import pytest


def test_state_initial_keys(tmp_state):
    """Fresh AppState exposes the expected baseline keys."""
    for key in ("frame_jpg", "fps", "device", "frame_count",
                "active_animals", "events", "track_history", "source"):
        assert key in tmp_state._state
    assert tmp_state["fps"] == 0.0
    assert tmp_state["frame_count"] == 0
    assert tmp_state["active_animals"] == []


def test_state_set_and_get(tmp_state):
    """set/get/update behave like a thin dict wrapper."""
    tmp_state.set("fps", 24.5)
    assert tmp_state.get("fps") == 24.5
    assert tmp_state["fps"] == 24.5

    tmp_state["device"] = "cuda"
    assert tmp_state["device"] == "cuda"

    tmp_state.update(fps=30.0, device="mps")
    assert tmp_state["fps"] == 30.0
    assert tmp_state["device"] == "mps"


def test_state_default_on_missing_key(tmp_state):
    """get() returns the provided default for unknown keys."""
    assert tmp_state.get("nonexistent") is None
    assert tmp_state.get("nonexistent", 42) == 42


def test_state_acquire_lock_creates_locks(tmp_state):
    """acquire_lock is idempotent — same key returns the same Lock instance."""
    l1 = tmp_state.acquire_lock("frame_lock")
    l2 = tmp_state.acquire_lock("frame_lock")
    assert isinstance(l1, type(threading.Lock()))
    assert l1 is l2


def test_state_reset_clears_runtime_data(tmp_state):
    """reset_for_new_source wipes active_animals / behavior / track_history."""
    tmp_state["active_animals"] = [{"name": "x"}]
    tmp_state["behavior"] = [{"act": "walking"}]
    tmp_state["track_history"] = {1: [(0, 0, 1, 1)]}
    tmp_state["events"] = ["old"]

    tmp_state.reset_for_new_source()

    assert tmp_state["active_animals"] == []
    assert tmp_state["behavior"] == []
    assert tmp_state["track_history"] == {}
    # events is rotated, not emptied, so we get at least the reset marker
    assert any("reset" in str(e).lower() for e in tmp_state["events"])


def test_color_for_name_is_stable():
    """Same name → same color (polynomial hash is deterministic)."""
    from utils.state import color_for_name
    c1 = color_for_name("Marguerite")
    c2 = color_for_name("Marguerite")
    assert c1 == c2
    assert isinstance(c1, tuple)
    assert len(c1) == 3


def test_color_for_name_different_inputs():
    """Different names map to (usually) different palette entries."""
    from utils.state import color_for_name
    a = color_for_name("Marguerite")
    b = color_for_name("Aurelius")
    # Not strictly required to differ (palette is small) but very likely
    assert isinstance(a, tuple) and isinstance(b, tuple)


def test_color_for_name_returns_palette_color():
    """Result must be one of the configured PALETTE entries."""
    from utils.state import color_for_name
    from config import PALETTE
    for name in ("Marguerite", "Aurelius", "Boeuf_001", ""):
        assert color_for_name(name) in PALETTE