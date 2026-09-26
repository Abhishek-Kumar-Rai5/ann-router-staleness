"""Phase 5' helpers: vecs I/O, a Python copy of the C++ seeded shuffle, and the
insertion/deletion orders.
"""

import numpy as np

M64 = 0xFFFFFFFFFFFFFFFF


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    assert (rows[:, 0] == d).all()
    return rows[:, 1:].copy().view(np.float32)


def read_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def write_fvecs(path, m):
    m = np.ascontiguousarray(m, dtype=np.float32)
    out = np.empty((m.shape[0], m.shape[1] + 1), dtype=np.int32)
    out[:, 0] = m.shape[1]
    out[:, 1:] = m.view(np.int32)
    out.tofile(path)


class MT64:

    def __init__(self, seed):
        self.mt = [0] * 312
        self.mt[0] = seed & M64
        for i in range(1, 312):
            self.mt[i] = (6364136223846793005 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) & M64
        self.i = 312

    def __call__(self):
        if self.i >= 312:
            for k in range(312):
                x = (self.mt[k] & 0xFFFFFFFF80000000) | (self.mt[(k + 1) % 312] & 0x7FFFFFFF)
                xa = x >> 1
                if x & 1:
                    xa ^= 0xB5026F5AA96619E9
                self.mt[k] = self.mt[(k + 156) % 312] ^ xa
            self.i = 0
        x = self.mt[self.i]
        self.i += 1
        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= x >> 43
        return x & M64


# Must produce exactly the same permutation as ars::MakeQuerySplit.
def fisher_yates(n, seed):
    r = MT64(seed)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        b = i + 1
        th = ((1 << 64) - b) % b
        while True:
            v = r()
            if v >= th:
                j = v % b
                break
        p[i], p[j] = p[j], p[i]
    return np.array(p, dtype=np.int64)


def id_order(n_pool, seed):
    return fisher_yates(n_pool, seed)


def ood_order(pool, counts, id_perm, anchor_seed):
    a = int(np.random.default_rng(anchor_seed).integers(0, len(pool)))
    p64 = pool.astype(np.float64)
    d = np.sqrt(((p64 - p64[a]) ** 2).sum(1))
    by_dist = np.lexsort((np.arange(len(pool)), d))
    rank_in_id = np.empty(len(pool), dtype=np.int64)
    rank_in_id[id_perm] = np.arange(len(pool))
    order, prev = [], 0
    for c in counts:
        inc = by_dist[prev:c]
        order.extend(inc[np.argsort(rank_in_id[inc], kind="stable")].tolist())
        prev = c
    return np.array(order, dtype=np.int64), a, d


def write_order(path, rows):
    with open(path, "w") as f:
        f.write("row\n")
        f.writelines(f"{int(r)}\n" for r in rows)


def sift_pool(learn, queries, exclude_rows):
    qset = {q.tobytes() for q in queries}
    seen, rows = set(), []
    for i, v in enumerate(learn):
        b = v.tobytes()
        if b in qset or b in seen:
            continue
        seen.add(b)
        rows.append(i)
    excl = set(int(r) for r in exclude_rows)
    rows = np.array([r for r in rows if r not in excl], dtype=np.int64)
    return learn[rows], rows
