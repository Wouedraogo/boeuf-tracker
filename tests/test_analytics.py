"""
test_analytics.py — AnalyticsState aggregation logic.
"""
import pytest


def _sample(name, key, breed, behavior, source="video1.mp4", frame=0):
    return {
        "proper_name": name,
        "key": key,
        "breed": breed,
        "behavior": behavior,
        "source": source,
        "frame": frame,
    }


def test_compute_profiles_empty_state():
    from analytics import AnalyticsState
    s = AnalyticsState()
    profiles = s.compute_profiles()
    assert profiles == {}


def test_compute_profiles_aggregates_per_animal():
    from analytics import AnalyticsState
    s = AnalyticsState()
    s.detection_samples = [
        _sample("Marguerite", "Boeuf_001", "Holstein", "grazing", "v1.mp4", 10),
        _sample("Marguerite", "Boeuf_001", "Holstein", "grazing", "v1.mp4", 20),
        _sample("Marguerite", "Boeuf_001", "Holstein", "walking", "v2.mp4", 5),
        _sample("Aurelius",   "Boeuf_002", "Charolais", "lying",   "v1.mp4", 30),
    ]
    profiles = s.compute_profiles()
    assert set(profiles.keys()) == {"Marguerite", "Aurelius"}

    marg = profiles["Marguerite"]
    assert marg["total_samples"] == 3
    assert marg["video_count"] == 2
    assert marg["videos"] == {"v1.mp4": 2, "v2.mp4": 1}
    assert marg["activities_pct"]["grazing"] == pytest.approx(66.7, abs=0.1)
    assert marg["activities_pct"]["walking"] == pytest.approx(33.3, abs=0.1)
    # first_frame = min frame across all samples, last_frame = max
    assert marg["first_frame"] == 5
    assert marg["last_frame"] == 20
    assert marg["breed"] == "Holstein"


def test_compute_profiles_skips_indeterminate_breed():
    """If all breeds seen for an animal are 'Indeterminate', that label sticks."""
    from analytics import AnalyticsState
    s = AnalyticsState()
    s.detection_samples = [
        _sample("X", "Boeuf_001", "Indeterminate", "grazing"),
        _sample("X", "Boeuf_001", "Indeterminate", "walking"),
    ]
    # After the skip rule, breed remains "Indeterminee" (default)
    prof = s.compute_profiles()["X"]
    assert prof["breed"] == "Indeterminee"


def test_compute_profiles_uses_last_known_breed():
    from analytics import AnalyticsState
    s = AnalyticsState()
    s.detection_samples = [
        _sample("X", "Boeuf_001", "Indeterminate", "grazing"),
        _sample("X", "Boeuf_001", "Salers",       "grazing"),
    ]
    prof = s.compute_profiles()["X"]
    assert prof["breed"] == "Salers"


def test_to_dict_clamps_fps_history():
    from analytics import AnalyticsState
    s = AnalyticsState()
    # Inject 500 entries — to_dict should trim to the last 300.
    s.fps_history = [{"t": i, "fps": 25.0} for i in range(500)]
    out = s.to_dict()
    assert len(out["fps_history"]) == 300
    assert out["fps_history"][-1]["t"] == 499


def test_to_dict_exposes_race_and_activity_counts():
    from analytics import AnalyticsState
    s = AnalyticsState()
    s.race_counts = {"Holstein": 5, "Charolais": 2}
    s.activity_counts = {"grazing": 100, "walking": 40}
    out = s.to_dict()
    assert out["race_counts"] == {"Holstein": 5, "Charolais": 2}
    assert out["activity_counts"]["grazing"] == 100


def test_compute_profiles_handles_missing_fields():
    """Robustness: samples without optional fields don't crash."""
    from analytics import AnalyticsState
    s = AnalyticsState()
    s.detection_samples = [
        {"proper_name": "X", "key": "Boeuf_001", "behavior": "grazing"},
        {"proper_name": "X", "key": "Boeuf_001", "source": "v1.mp4"},
        {},  # fully empty sample — must be ignored or routed to '?'
    ]
    profiles = s.compute_profiles()
    assert "X" in profiles
    assert profiles["X"]["total_samples"] >= 2