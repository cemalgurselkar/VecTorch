import numpy as np

from vectorch import Vectorch


def test_cosine_ranking(tmp_path):
    db = Vectorch(tmp_path)
    c = db.create_collection("test", dimension=2, metric="cosine")

    c.add("same", np.array([1.0, 0.0], dtype=np.float32))
    c.add("close", np.array([0.8, 0.2], dtype=np.float32))
    c.add("opposite", np.array([-1.0, 0.0], dtype=np.float32))

    results = c.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=3,
    )

    assert [r.id for r in results] == ["same", "close", "opposite"]


def test_dot_ranking(tmp_path):
    db = Vectorch(tmp_path)
    c = db.create_collection("test", dimension=2, metric="dot")

    c.add("a", np.array([1.0, 0.0], dtype=np.float32))
    c.add("b", np.array([2.0, 0.0], dtype=np.float32))
    c.add("c", np.array([-1.0, 0.0], dtype=np.float32))

    results = c.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=3,
    )

    assert [r.id for r in results] == ["b", "a", "c"]


def test_l2_ranking(tmp_path):
    db = Vectorch(tmp_path)
    c = db.create_collection("test", dimension=2, metric="l2")

    c.add("nearest", np.array([1.0, 1.0], dtype=np.float32))
    c.add("middle", np.array([2.0, 2.0], dtype=np.float32))
    c.add("far", np.array([10.0, 10.0], dtype=np.float32))

    results = c.search(
        np.array([0.0, 0.0], dtype=np.float32),
        k=3,
    )

    assert [r.id for r in results] == ["nearest", "middle", "far"]