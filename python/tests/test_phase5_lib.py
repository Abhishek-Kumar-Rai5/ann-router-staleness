import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase5_lib as L  # noqa: E402


def test_fisher_yates_matches_cpp_golden_and_is_a_permutation():
    p = L.fisher_yates(20, 1)
    assert sorted(p[:5].tolist()) == [1, 7, 10, 14, 17]
    q = L.fisher_yates(1000, 7)
    assert sorted(q.tolist()) == list(range(1000))
    assert (L.fisher_yates(1000, 7) == q).all()


def test_ood_order_is_nested_balls_in_id_order():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(500, 4)).astype(np.float32)
    idp = L.fisher_yates(500, 3)
    counts = [20, 50, 120]
    order, a, d = L.ood_order(pool, counts, idp, 11)
    assert len(order) == 120 and len(set(order.tolist())) == 120
    by_dist = np.lexsort((np.arange(500), d))
    rank = np.empty(500, int)
    rank[idp] = np.arange(500)
    prev = 0
    for c in counts:
        inc = order[prev:c]
        assert set(inc.tolist()) == set(by_dist[prev:c].tolist())
        assert (np.diff(rank[inc]) > 0).all()
        prev = c
    assert d[a] == 0


def test_fvecs_roundtrip(tmp_path):
    m = np.random.default_rng(1).normal(size=(7, 3)).astype(np.float32)
    L.write_fvecs(tmp_path / "x.fvecs", m)
    assert np.array_equal(L.read_fvecs(tmp_path / "x.fvecs"), m)


def test_sift_pool_excludes_query_copies_dedups_and_drops_rows():
    learn = np.array([[1, 1], [2, 2], [1, 1], [3, 3], [4, 4], [5, 5]], dtype=np.float32)
    queries = np.array([[3, 3]], dtype=np.float32)
    vec, rows = L.sift_pool(learn, queries, exclude_rows=[4])
    assert rows.tolist() == [0, 1, 5]
    assert np.array_equal(vec, learn[[0, 1, 5]])
