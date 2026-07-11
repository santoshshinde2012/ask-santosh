import pytest

from ask_santosh.similarity import cosine


def test_identical_vectors_are_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_are_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_zero_vector_is_zero_not_error():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])
