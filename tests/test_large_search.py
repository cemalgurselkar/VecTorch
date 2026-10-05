import numpy as np

from vectorch import Vectorch


def test_large_matrix_search(tmp_path):
    rng = np.random.default_rng(42)

    n = 10_000
    dimension = 128

    vectors = rng.random(
        (n, dimension),
        dtype=np.float32,
    )

    db = Vectorch(tmp_path)
    c = db.create_collection(
        "large",
        dimension=dimension,
        metric="l2",
    )

    for i in range(n):
        c.add(i, vectors[i])

    query = vectors[537]

    results = c.search(query, k=10)

    assert len(results) == 10

    # Query itself must be the nearest vector.
    assert results[0].id == 537
    assert np.isclose(results[0].score, 0.0)