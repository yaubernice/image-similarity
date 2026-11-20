import math

from metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k, summarise


def test_precision_and_recall():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 1.0


def test_average_precision():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    expected = ((1 / 1) + (2 / 3)) / 2
    assert abs(average_precision(retrieved, relevant) - expected) < 1e-9


def test_ndcg_binary():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    dcg = 1.0 / math.log2(2) + 0 + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(ndcg_at_k(retrieved, relevant, 3) - dcg / idcg) < 1e-9


def test_empty_relevant_is_zero():
    assert precision_at_k(["a"], set(), 1) == 0.0
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert average_precision(["a"], set()) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_summarise_means():
    rankings = [
        {"retrieved": ["a", "b"], "relevant": {"a"}},
        {"retrieved": ["b", "a"], "relevant": {"a"}},
    ]
    summary = summarise(rankings, ks=(1,))
    assert summary["queries"] == 2
    assert summary["precision@1"] == 0.5
    assert summary["mAP"] == round((1.0 + 0.5) / 2, 4)
