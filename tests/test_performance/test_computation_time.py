import time
from isomorphic.extractors import MeanPoolingExtractor

def test_extraction_is_fast():
    e = MeanPoolingExtractor("dummy", embedding_dim=128)
    start = time.time()
    e.extract_batch(["hello world"] * 100)
    assert time.time() - start < 2
