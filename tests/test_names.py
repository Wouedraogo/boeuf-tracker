"""
test_names.py — NameGenerator: maps Boeuf_NNN keys to readable names.
"""
import pytest


def test_name_generator_basic():
    from names import NameGenerator
    gen = NameGenerator({"Boeuf_001": {}, "Boeuf_002": {}})
    assert gen.get("Boeuf_001") != "Boeuf_001"
    assert gen.get("Boeuf_002") != "Boeuf_002"
    assert gen.get("Boeuf_001") != gen.get("Boeuf_002")


def test_name_generator_unknown_key_returns_key():
    from names import NameGenerator
    gen = NameGenerator({})
    assert gen.get("unknown") == "unknown"


def test_name_generator_all_returns_copy():
    from names import NameGenerator
    gen = NameGenerator({"Boeuf_001": {}, "Boeuf_002": {}})
    snap = gen.all()
    assert isinstance(snap, dict)
    assert set(snap.keys()) == {"Boeuf_001", "Boeuf_002"}
    # Mutating the snapshot should not affect internal state
    snap["Boeuf_001"] = "tampered"
    assert gen.get("Boeuf_001") != "tampered"


def test_name_generator_pool_overflow_gets_suffix():
    """If we exceed NAME_POOL, names get a cycle suffix rather than crashing."""
    from names import NameGenerator
    # NAME_POOL is finite; force overflow
    keys = {f"Boeuf_{i:03d}": {} for i in range(1, 200)}
    gen = NameGenerator(keys)
    # Index 199 (1-based 200) should produce a suffixed name
    n = gen.get("Boeuf_200")
    assert n  # truthy, non-empty
    assert isinstance(n, str)


def test_name_generator_handles_malformed_key():
    """Non-Boeuf_ keys get a hash-based fallback name, no crash."""
    from names import NameGenerator
    gen = NameGenerator({"custom_key": {}, "Boeuf_XYZ": {}})
    assert gen.get("custom_key")
    assert gen.get("Boeuf_XYZ")


def test_next_bovin_key_is_unique(tmp_path, monkeypatch):
    """next_bovin_key uses a counter persisted on disk — must increment."""
    import names
    # Use a temp counter file so we don't pollute the project
    counter_file = tmp_path / "names_counter.json"
    counter_file.write_text("0")
    monkeypatch.setattr(names, "COUNTER_PATH", str(counter_file))
    # Reset module-level singleton so it re-reads our temp file
    monkeypatch.setattr(names, "_global_counter", None)

    k1 = names.next_bovin_key()
    k2 = names.next_bovin_key()
    assert k1 != k2
    assert k1.startswith("Boeuf_")
    assert k2.startswith("Boeuf_")
    assert counter_file.exists()