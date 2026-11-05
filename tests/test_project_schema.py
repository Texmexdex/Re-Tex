"""FEAT-005 — project schema is ADDITIVE.

The autonomous-output path may attach a per-track `automation` lane (design D / B.4). These tests prove:
  * track(automation=[...]) round-trips the lane verbatim through JSON;
  * a track / project with NO automation serializes BYTE-IDENTICALLY to the pre-change output — the
    `automation` key is simply absent, so a Phase-2-free project opened in the manual timeline is unchanged.

project.empty() carries a random id + a `created` timestamp, so the byte-identical assertions neutralize
those two fields (they were already random/time-based before FEAT-005) and compare everything else exactly.
"""
import json

import mashlab.project as project


# The exact per-track field set + ORDER the manual flow produced before FEAT-005 (project.track with no
# automation). If the schema ever drifts, this literal is what a Phase-2-free export is compared against.
PRE_CHANGE_TRACK_KEYS = ["id", "name", "role", "color", "gain_db", "pan", "mute", "solo",
                         "hpf", "lpf", "duck", "pocket"]


def _canonical(d):
    """Serialize with id/created neutralized so two independent empty() calls compare for everything but the
    intentionally-random identity + timestamp."""
    d = json.loads(json.dumps(d))
    d["id"] = "<id>"
    d.pop("created", None)
    d.pop("modified", None)
    for t in d.get("tracks", []):
        t["id"] = "<tid>"
    return json.dumps(d, sort_keys=False)


def test_track_no_automation_is_byte_identical_shape():
    t = project.track("Drums", "drums")
    # the ADDITIVE field must be entirely ABSENT when not supplied
    assert "automation" not in t
    assert list(t.keys()) == PRE_CHANGE_TRACK_KEYS
    # and None behaves the same as omitting it
    t_none = project.track("Drums", "drums")
    assert "automation" not in t_none
    assert list(t_none.keys()) == PRE_CHANGE_TRACK_KEYS


def test_track_automation_round_trips():
    lane = [
        {"kind": "sweep", "type": 3, "beat": 16, "from_hz": 200.0, "to_hz": 8000.0, "curve": "exp"},
        {"kind": "noise_sweep", "type": 6, "beat": 48, "from_hz": 8000.0, "to_hz": 200.0, "curve": "lin"},
    ]
    t = project.track("Vocals", "vocals", automation=lane)
    assert t["automation"] == lane
    # survives a JSON round-trip unchanged
    again = json.loads(json.dumps(t))
    assert again["automation"] == lane
    # it is a copy, not an alias of the caller's list
    lane.append({"kind": "x"})
    assert len(t["automation"]) == 2


def test_empty_track_automation_absent_everywhere():
    p = project.empty()
    assert all("automation" not in t for t in p["tracks"])


def test_empty_project_byte_identical_across_calls():
    a = project.empty(name="X", bpm=120.0, bpb=4)
    b = project.empty(name="X", bpm=120.0, bpb=4)
    # identical except the deliberately random id + created timestamp
    assert _canonical(a) == _canonical(b)
    # no Phase-2 field leaked into the default serialization
    blob = json.dumps(a)
    assert "automation" not in blob


def test_empty_project_matches_pre_change_structure():
    """The full default project still has exactly the pre-FEAT-005 top-level keys and track shape."""
    p = project.empty()
    assert set(p.keys()) == {"id", "name", "bpm", "bpb", "quality", "songs", "tracks", "clips",
                             "master", "loop", "created", "notes"}
    assert [t["role"] for t in p["tracks"]] == ["drums", "bass", "other", "vocals"]
    for t in p["tracks"]:
        assert list(t.keys()) == PRE_CHANGE_TRACK_KEYS
