import numpy as np
import pytest

@pytest.fixture
def sample_texts():
    return ["sample toxic sentence", "another sample sentence"]

@pytest.fixture
def sample_matrices():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (32, 16)), rng.normal(0, 1, (32, 16))
