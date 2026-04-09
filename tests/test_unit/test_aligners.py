from isomorphic.alignment_utils import ProcrustesAligner

def test_align_shape(sample_matrices):
    x, y = sample_matrices
    q, err = ProcrustesAligner.align(x, y)
    assert q.shape == (x.shape[1], x.shape[1])
    assert err >= 0
