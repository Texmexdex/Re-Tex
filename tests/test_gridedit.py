"""web/js/gridedit.js (run in node): the beat grid window's editing rules, as the user set them (Oct 19) - a bar's
beats always even between its bar lines unless placed by hand, nothing before the bar being edited ever moves."""
import json
import shutil
import subprocess

import pytest

from mashlab.config import WEB

JS = r"""
import * as E from "./gridedit.mjs";
const grid = () => ({ t: [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], d: [1,0,0,0, 1,0,0,0, 1,0,0,0, 1],
                      s: Array(13).fill(1), c: Array(13).fill(0), p0: null });
const base = (M) => ({ t: M.t.slice(), c: M.c.slice() });
const r = (M) => M.t.map((x) => Math.round(x * 1000) / 1000);
const out = {};
// the user's case: three beats bunched in the back half of a bar; dragging the next bar line evens it out
let M = grid(); M.t.splice(4, 5, 2, 2.82, 3.22, 3.62, 4.0);
out.bunched = E.unevenBars(M);
E.dragMark(M, 8, 4.2, "plain", base(M)); out.dragBarLine = r(M);
M = grid(); E.dragMark(M, 4, 2.2, "one", base(M)); out.ctrlBarLine = r(M);
M = grid(); E.dragMark(M, 5, 2.6, "plain", base(M)); out.dragBeat = r(M);
M = grid(); E.dragMark(M, 6, 3.1, "one", base(M)); out.ctrlBeat = r(M); out.ctrlBeatPlaced = M.c[6];
E.dragMark(M, 8, 4.4, "plain", base(M)); out.placedBarKeepsShape = r(M);
M = grid(); E.dragMark(M, 6, 3.3, "all", base(M)); out.shiftAll = r(M);
M = grid(); E.deleteMark(M, 5); out.deleteBeat = r(M); out.deleteBeatDowns = M.d;
M = grid(); E.deleteMark(M, 8); out.deleteBarLine = r(M);
M = grid(); E.deleteMark(M, 5); out.addBack = E.addMark(M, 2.3); out.addBackT = r(M);
M = grid(); out.addExact = E.addMark(M, 2.3, true); out.addExactT = r(M);
M = grid(); E.toggleDown(M, 6); out.toggle = M.d;
out.evenGrid = E.unevenBars(grid());
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    if not shutil.which("node"):
        pytest.skip("needs node")
    d = tmp_path_factory.mktemp("gridedit")
    shutil.copy(WEB / "js" / "gridedit.js", d / "gridedit.mjs")
    (d / "run.mjs").write_text(JS, encoding="utf-8")
    return json.loads(subprocess.run(["node", str(d / "run.mjs")], capture_output=True, text=True, check=True).stdout)


def close(a, b):
    return len(a) == len(b) and all(abs(x - y) < 2e-3 for x, y in zip(a, b))


def test_a_bunched_bar_is_found_and_evened_by_dragging_its_next_bar_line(res):
    assert res["bunched"] == [[4, 8]]
    # bar 1 untouched; bar 2 spread evenly from 2.0 to 4.2; everything after moved along by +0.2
    assert close(res["dragBarLine"], [0, .5, 1, 1.5, 2, 2.55, 3.1, 3.65, 4.2, 4.7, 5.2, 5.7, 6.2])


def test_ctrl_drag_bar_line_changes_only_the_bars_either_side(res):
    assert close(res["ctrlBarLine"], [0, .55, 1.1, 1.65, 2.2, 2.65, 3.1, 3.55, 4, 4.5, 5, 5.5, 6])


def test_drag_a_beat_changes_its_bars_tempo_and_moves_the_rest_along(res):
    assert close(res["dragBeat"], [0, .5, 1, 1.5, 2, 2.6, 3.2, 3.8, 4.4, 4.9, 5.4, 5.9, 6.4])


def test_ctrl_drag_a_beat_moves_only_it_and_its_bar_keeps_that_shape(res):
    assert close(res["ctrlBeat"], [0, .5, 1, 1.5, 2, 2.5, 3.1, 3.5, 4, 4.5, 5, 5.5, 6]) and res["ctrlBeatPlaced"] == 1
    # dragging the next bar line afterwards stretches that bar in proportion - the placed beat keeps its place in it
    assert close(res["placedBarKeepsShape"], [0, .5, 1, 1.5, 2, 2.6, 3.32, 3.8, 4.4, 4.9, 5.4, 5.9, 6.4])


def test_shift_drag_moves_everything(res):
    assert close(res["shiftAll"], [x + .3 for x in [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]])


def test_deleting_a_beat_respreads_its_bar_and_deleting_a_bar_line_joins_two_bars(res):
    assert close(res["deleteBeat"], [0, .5, 1, 1.5, 2, 2 + 2 / 3, 2 + 4 / 3, 4, 4.5, 5, 5.5, 6])        # 4 beats -> 3
    assert res["deleteBeatDowns"] == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    assert close(res["deleteBarLine"], [0, .5, 1, 1.5, 2] + [2 + k * 4 / 7 for k in range(1, 7)] + [6])  # 7 even beats


def test_adding_a_beat_respreads_its_bar_unless_placed_exactly(res):
    assert res["addBack"] == 5 and close(res["addBackT"], [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    assert res["addExact"] == 5 and close(res["addExactT"], [0, .5, 1, 1.5, 2, 2.3, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])


def test_double_click_makes_a_beat_a_bars_one_and_an_even_grid_has_no_uneven_bars(res):
    assert res["toggle"] == [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    assert res["evenGrid"] == []
