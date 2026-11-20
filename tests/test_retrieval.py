import numpy as np

from retrieval import apply_whitening, expand_query, fit_whitening, l2_normalize


def test_whitening_reduces_dimension_and_normalises():
    rng = np.random.default_rng(0)
    features = l2_normalize(rng.normal(size=(40, 16)).astype(np.float32))
    state = fit_whitening(features, out_dim=8)
    transformed = apply_whitening(features, state)
    assert transformed.shape == (40, 8)
    norms = np.linalg.norm(transformed, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_query_expansion_is_unit_length():
    query = np.array([1.0, 0.0], dtype=np.float32)
    neighbours = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    expanded = expand_query(query, neighbours, np.array([0.2, 0.9], dtype=np.float32), alpha=3)
    assert expanded.shape == (2,)
    assert abs(np.linalg.norm(expanded) - 1.0) < 1e-5
