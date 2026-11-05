"""timing.detected_marks (Python) is the twin of web/js/timing.js detectedMarks: when a song's grid changes, the old
projects are frozen on the detector's marks as the page showed them - the same beats, bar lines and sure / unsure."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from mashlab import timing
from mashlab.config import WEB


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_python_and_js_detected_marks_agree(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    beats = list(np.round(np.cumsum(rng.uniform(0.45, 0.55, n)) - 1.0, 5))       # a few before 0 and past the end
    info = dict(beats=beats, bars=list(range(2, n, 4)), bpb=4, duration=float(beats[-3]), first_beat=4, last_beat=110,
                beat_conf=[int(x) for x in rng.integers(0, 100, n)],
                mus_check={"v": [int(x) for x in rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])]})
    shutil.copy(WEB / "js" / "timing.js", tmp_path / "timing.mjs")
    (tmp_path / "run.mjs").write_text('import { detectedMarks } from "./timing.mjs";\n'
                                      f"console.log(JSON.stringify(detectedMarks({json.dumps(info)})));\n", encoding="utf-8")
    r = subprocess.run(["node", str(tmp_path / "run.mjs")], capture_output=True, text=True)
    if r.returncode != 0 and "Cannot find module" in r.stderr:
        pytest.skip("timing.js imports another module")
    js = json.loads(r.stdout)
    py = timing.detected_marks(info)
    assert py["t"] == js["t"] and py["d"] == js["d"] and py["bpb"] == js["bpb"]
    assert py["s"] == [int(x) for x in (js["s"] if isinstance(js["s"], list) else [js["s"][str(k)] for k in range(len(js["s"]))])]
