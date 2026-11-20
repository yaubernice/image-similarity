import faiss
import numpy as np

from image_similarity import build_faiss_index, load_index, save_index


def test_ip_index_roundtrip(tmp_path):
    features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        dtype=np.float32,
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    index = build_faiss_index(features, metric="ip")
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"]
    save_index(index, paths, features, tmp_path / "index", model="unit", metric="ip")

    loaded, loaded_paths, loaded_features = load_index(tmp_path / "index")
    assert loaded.ntotal == 3
    assert loaded_paths == paths
    np.testing.assert_allclose(loaded_features, features)

    query = features[:1]
    scores, ids = loaded.search(query, 2)
    assert int(ids[0, 0]) == 0
    assert scores[0, 0] > scores[0, 1]
