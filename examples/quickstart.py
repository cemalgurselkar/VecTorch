from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from vectorch import Vectorch


def main() -> None:
    default_path = Path(tempfile.gettempdir()) / "vectorch-quickstart"
    database_path = Path(os.environ.get("VECTORCH_DB_PATH", default_path))

    with Vectorch(database_path) as db:
        if "documents" in db.list_collections():
            documents = db.get_collection("documents")
        else:
            documents = db.create_collection(
                "documents",
                dimension=3,
                metric="cosine",
            )
            documents.add(
                "python",
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                {"title": "Python"},
            )
            documents.add(
                "database",
                np.array([0.8, 0.2, 0.0], dtype=np.float32),
                {"title": "Databases"},
            )

        results = documents.search(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            k=2,
        )

        print(f"Database: {database_path}")
        for result in results:
            print(
                f"id={result.id!r} score={result.score:.4f} "
                f"metadata={result.metadata!r}"
            )


if __name__ == "__main__":
    main()
