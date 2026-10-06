import numpy as np
import pytest

from vectorch.validation import validate_vector


def test_accepts_float32_numpy_without_copy():
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = validate_vector(vector, dimension=3)

    assert result.dtype == np.float32
    assert result.shape == (3,)
    assert result.flags.c_contiguous
    assert result is vector


def test_converts_list_to_float32():
    result = validate_vector(
        [1.0, 2.0, 3.0],
        dimension=3,
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (3,)


def test_converts_float64_to_float32():
    vector = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    result = validate_vector(vector, dimension=3)

    assert result.dtype == np.float32
    assert np.allclose(result, vector)


def test_rejects_wrong_dimension():
    vector = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError):
        validate_vector(vector, dimension=3)


def test_rejects_2d_input():
    vector = np.array(
        [[1.0, 2.0, 3.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_vector(vector, dimension=3)


@pytest.mark.parametrize(
    "vector",
    [
        [1.0, np.nan, 3.0],
        [1.0, np.inf, 3.0],
        [1.0, -np.inf, 3.0],
    ],
)
def test_rejects_non_finite_values(vector):
    with pytest.raises(ValueError):
        validate_vector(vector, dimension=3)


def test_returns_contiguous_array():
    source = np.arange(6, dtype=np.float32)
    vector = source[::2]

    assert not vector.flags.c_contiguous

    result = validate_vector(vector, dimension=3)

    assert result.flags.c_contiguous
    assert np.array_equal(result, vector)


def test_accepts_cpu_torch_tensor_zero_copy():
    torch = pytest.importorskip("torch")

    tensor = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
    )

    result = validate_vector(tensor, dimension=3)

    assert result.dtype == np.float32
    assert result.shape == (3,)
    assert result.flags.c_contiguous
    assert np.shares_memory(result, tensor.numpy())