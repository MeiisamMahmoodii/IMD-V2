import numpy as np
from scipy import stats

def test_dummy_significance():
    rng = np.random.default_rng(42)
    a = rng.normal(0.7, 0.05, 200)
    b = rng.normal(0.6, 0.05, 200)
    _, p = stats.ttest_ind(a, b)
    assert p < 0.05
