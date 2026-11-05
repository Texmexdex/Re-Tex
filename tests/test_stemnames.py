"""web/js/stemnames.js (run in node): a folder holding several songs' stems is split into one song per set, each file
on the right stem, and whole songs lying next to them are left out - checked on the Radiohead remix-contest names."""
import json
import shutil
import subprocess

import pytest

from mashlab.config import WEB

NAMES = ["01 Reckoner (Backing Vocals Stem).wav", "02 Reckoner (Bass Stem).wav", "03 Reckoner (Drums _ Percussion Stem).wav",
         "04 Reckoner (Guitars Stem).wav", "05 Reckoner (Lead Vocal Stem).wav", "06 Reckoner (Piano _ Strings Stem).wav",
         "07. Motion Picture Soundtrack.mp3", "Nude (Bass Stem).wav", "Nude (Drum Stem).wav", "Nude (Guitar Stem).wav",
         "Nude (String FX Etc. S.wav", "Nude (Voice Stem).wav", "Street Spirit (Live).mp3",
         "Beep Street - Kick and Snare.wav", "Beep Street - Hats.wav", "1_MySong_(Vocals).wav", "1_MySong_(Instrumental).wav"]


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_songs_and_stems_from_file_names(tmp_path):
    shutil.copy(WEB / "js" / "stemnames.js", tmp_path / "stemnames.mjs")
    (tmp_path / "run.mjs").write_text(
        'import { groupSongs, startRoles } from "./stemnames.mjs";\n'
        f"const names = {json.dumps(NAMES)};\n"
        'const g = groupSongs(names, "Radiohead"), r = startRoles(names, g);\n'
        "console.log(JSON.stringify(g.map((x) => [x.title, x.items.map((i) => [names[i], r[i]])])));\n", encoding="utf-8")
    out = subprocess.run(["node", str(tmp_path / "run.mjs")], capture_output=True, text=True, check=True).stdout
    songs = {t: dict(files) for t, files in json.loads(out)}
    assert songs["Reckoner"] == {
        "01 Reckoner (Backing Vocals Stem).wav": "vocals", "02 Reckoner (Bass Stem).wav": "bass",
        "03 Reckoner (Drums _ Percussion Stem).wav": "drums", "04 Reckoner (Guitars Stem).wav": "other",
        "05 Reckoner (Lead Vocal Stem).wav": "vocals", "06 Reckoner (Piano _ Strings Stem).wav": "other"}
    assert songs["Nude"] == {"Nude (Bass Stem).wav": "bass", "Nude (Drum Stem).wav": "drums", "Nude (Guitar Stem).wav": "other",
                             "Nude (String FX Etc. S.wav": "other", "Nude (Voice Stem).wav": "vocals"}
    assert songs["Motion Picture Soundtrack"] == {"07. Motion Picture Soundtrack.mp3": ""}     # a whole song: left out
    assert songs["Street Spirit"] == {"Street Spirit (Live).mp3": ""}
    assert songs["Beep Street"] == {"Beep Street - Kick and Snare.wav": "drums", "Beep Street - Hats.wav": "drums"}
    assert songs["MySong"] == {"1_MySong_(Vocals).wav": "vocals", "1_MySong_(Instrumental).wav": "other"}
