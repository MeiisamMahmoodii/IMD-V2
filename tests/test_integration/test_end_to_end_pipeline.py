from isomorphic.pipeline import IsomorphicPipeline

def test_end_to_end_alignment():
    pipe = IsomorphicPipeline("dummy")
    src = ["a b c", "d e f", "g h i"]
    tgt = ["a x c", "d y f", "g z i"]
    out = pipe.run_alignment(src, tgt, "exp_integration")
    assert "alignment" in out
