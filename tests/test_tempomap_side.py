"""The TempoMap sidecar: only its own files are served, and only for plain song ids."""
from mashlab import tempomap_side as T


def test_names():
    assert T.FILE_OK.match("report.html") and T.FILE_OK.match("mix_map.mp3") and T.FILE_OK.match("marks.json")
    for bad in ("../song.json", "report.html/../x", "mix_map.mp3.exe", "settings.json", ""):
        assert not T.FILE_OK.match(bad)
    assert T.SID_OK.match("Golem_62cf3f")
    for bad in ("../x", "a/b", "a b", ""):
        assert not T.SID_OK.match(bad)
    assert T.file_path("../x", "report.html") is None and T.file_path("Golem_62cf3f", "../x") is None


def test_midi_refused():
    import pytest
    with pytest.raises(ValueError):
        T.check_sid("midi_01_the_grudge_a46fce")
