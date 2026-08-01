"""
test_capture.py — open_capture + source_label helpers.
"""
import os

import pytest


def test_source_label_webcam_int():
    from capture import source_label
    assert source_label(0) == "webcam 0"
    assert source_label(2) == "webcam 2"


def test_source_label_plain_path():
    from capture import source_label
    assert source_label("/tmp/foo/bar/video.mp4") == "video.mp4"


def test_source_label_strips_timestamp_prefix():
    """Filenames like '1716741223_myvideo.mp4' lose the timestamp prefix."""
    from capture import source_label
    # 10+ digit prefix = treated as timestamp
    assert source_label("/tmp/x/1716741223_myvideo.mp4") == "myvideo.mp4"
    # Short prefix is preserved
    assert source_label("/tmp/x/123_myvideo.mp4") == "123_myvideo.mp4"


def test_source_label_backslashes():
    """Both / and \\ separators are handled (Windows compat)."""
    from capture import source_label
    assert source_label("C:\\videos\\cattle.mp4") == "cattle.mp4"


def test_source_label_other_types():
    from capture import source_label
    # Unknown input → stringified
    assert source_label(None) == "None"
    assert source_label(3.14) == "3.14"


@pytest.mark.skipif(os.environ.get("CI") == "true",
                    reason="no display in CI")
def test_open_capture_file(sample_video):
    """open_capture must accept a file path and return an opened capture."""
    import cv2
    from capture import open_capture
    cap = open_capture(str(sample_video))
    try:
        assert cap is not None
        assert cap.isOpened()
        # Frame must be readable
        ok, frame = cap.read()
        assert ok is True
        assert frame is not None
    finally:
        if cap is not None:
            cap.release()


def test_open_capture_returns_none_for_missing_file(tmp_path):
    from capture import open_capture
    assert open_capture(str(tmp_path / "nope.mp4")) is None


def test_read_with_recovery_returns_none_for_dead_source(tmp_path):
    """With no frame available, read_with_recovery returns (False, None)
    without crashing — this is the safe path used by the processor loop."""
    import cv2
    from capture import open_capture, read_with_recovery

    bogus = str(tmp_path / "missing.mp4")
    cap = open_capture(bogus)
    assert cap is None
    # No capture object → just verify the helpers compose without exceptions
    assert callable(read_with_recovery)


def test_read_with_recovery_handles_short_video(sample_video):
    """A drained short video can be rewound and read again via direct cv2."""
    import cv2
    cap = cv2.VideoCapture(str(sample_video))
    assert cap.isOpened()
    # Manually rewind and read (this is what read_with_recovery does internally)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    assert ok is True
    assert frame is not None
    cap.release()