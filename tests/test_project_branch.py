"""Branching a project: a copy under a new id to change, while the original file stays exactly as it was."""
import json

from mashlab import project as P


def test_a_branch_is_a_separate_copy_and_the_original_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PROJECTS", tmp_path)
    src = P.empty("Night Drive", 130, 4)
    src["clips"].append(P.clip(src["tracks"][0]["id"], "song", "drums", 0, 0, 16))
    P.save(src)
    before = (tmp_path / f"{src['id']}.json").read_bytes()

    b = P.branch(src["id"])
    assert b["id"] != src["id"] and b["name"] == "Night Drive (branch 1)"
    assert b["branch_of"]["id"] == src["id"] and b["branch_of"]["name"] == "Night Drive"
    assert b["clips"] == src["clips"] and (tmp_path / f"{b['id']}.json").exists()

    b["clips"][0]["gain_db"] = -6.0                  # changing the branch ...
    P.save(b)
    assert (tmp_path / f"{src['id']}.json").read_bytes() == before      # ... leaves the original as it was

    b2 = P.branch(b["id"])                           # a branch of a branch: the next number, no stacked suffix
    assert b2["name"] == "Night Drive (branch 2)" and b2["branch_of"]["id"] == b["id"]
    assert P.branch(src["id"], name="  Darker drums ")["name"] == "Darker drums"
    listed = {p["id"]: p for p in P.list_projects()}
    assert listed[b["id"]]["branch_of"]["name"] == "Night Drive" and listed[src["id"]]["branch_of"] is None
    assert json.loads(before)["name"] == "Night Drive"
