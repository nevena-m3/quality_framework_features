from paper1_qc.segmentation import Interval, build_segmentation_views


def test_segmentation_views_are_distinct_and_guarded():
    raw = [Interval(0.10, 1.00), Interval(1.05, 2.00), Interval(3.00, 4.00)]
    views = build_segmentation_views(
        raw,
        duration_sec=5.0,
        bridge_gap_ms=100,
        min_speech_ms=250,
        strict_speech_edge_ms=50,
        strict_nonspeech_edge_ms=200,
    )
    assert len(views["raw_speech"]) == 3
    assert len(views["primary_speech"]) == 2
    assert views["primary_speech"][0] == Interval(0.10, 2.00)
    assert abs(views["strict_speech"][0].start_sec - 0.15) < 1e-12
    assert abs(views["strict_speech"][0].end_sec - 1.95) < 1e-12
    assert len(views["strict_internal_nonspeech"]) == 1
    assert abs(views["strict_internal_nonspeech"][0].start_sec - 2.20) < 1e-12
    assert abs(views["strict_internal_nonspeech"][0].end_sec - 2.80) < 1e-12
