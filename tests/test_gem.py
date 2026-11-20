import numpy as np

from extractors import gem_pool


def test_gem_pool_matches_a_hand_calculation():
    feature_map = np.array([[[[1.0, 8.0], [1.0, 1.0]]]], dtype=np.float32)
    # shape (1, 1, 2, 2) — mean of cubes, then cube root
    pooled = gem_pool(feature_map, p=3.0)
    expected_first = ((1.0 ** 3 + 1.0 ** 3) / 2) ** (1 / 3)
    expected_second = ((8.0 ** 3 + 1.0 ** 3) / 2) ** (1 / 3)
    np.testing.assert_allclose(pooled[0], [expected_first, expected_second], rtol=1e-5)
