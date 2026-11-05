"""Library grouping: file-name parsing, which value wins, and the 'works together' clustering."""
import numpy as np

from mashlab import libmeta


def test_from_filename():
    assert libmeta.from_filename("1790318015146_Puscifer  - Agostina [rWQwjAu752c].wav") == {"artist": "Puscifer", "title": "Agostina"}
    assert libmeta.from_filename("1790318572656_02 Lechium.mp3") == {"title": "Lechium"}
    assert libmeta.from_filename("Easy [YPmJVWJzNGY].wav") == {"title": "Easy"}


def test_user_beats_tags_beats_file_beats_guess():
    e = {"guess": {"genre": "Rock"}, "file": {"artist": "From File"}, "tags": {"artist": "Tagged", "genre": "Metal"},
         "user": {"genre": "Post-Hardcore"}}
    m = libmeta._effective(e)
    assert m["artist"] == "Tagged" and m["src"]["artist"] == "tags"
    assert m["genre"] == "Post-Hardcore" and m["src"]["genre"] == "user"
    assert libmeta._effective({"guess": {"genre": "Rock"}})["src"]["genre"] == "guess"


def test_cluster_groups_songs_that_fit_each_other():
    M = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (1, 2)]:          # 0-1-2 all fit (80)
        M[a, b] = M[b, a] = 80
    M[3, 4] = M[4, 3] = 75                         # 3-4 fit
    M[2, 3] = M[3, 2] = 50                         # weak link between the groups
    groups = sorted(sorted(g) for g in libmeta.cluster(M, min_fit=62))
    assert groups == [[0, 1, 2], [3, 4], [5]]


def test_cluster_respects_max_size():
    M = np.full((5, 5), 90.0)
    np.fill_diagonal(M, 0)
    assert max(len(g) for g in libmeta.cluster(M, min_fit=62, max_size=3)) <= 3
