from isomorphic.extractors import MeanPoolingExtractor, LastTokenExtractor, HybridExtractor

def test_extract_shapes(sample_texts):
    for cls in [MeanPoolingExtractor, LastTokenExtractor, HybridExtractor]:
        e = cls("dummy", embedding_dim=64)
        assert e.extract(sample_texts[0]).shape == (64,)
