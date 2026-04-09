from isomorphic.pipeline import IsomorphicPipeline


def test_build_final_dataset_threshold_applies(tmp_path):
    pipeline = IsomorphicPipeline("dummy", output_root=tmp_path)
    source = ["Alpha text", "Beta text"]
    candidates = ["Alpha text", "Completely different text"]
    stats = pipeline.build_final_dataset(
        source_texts=source,
        candidate_texts=candidates,
        threshold=0.5,
        output_path=str(tmp_path / "final_dataset.json"),
    )
    assert stats["total_rows"] == 2
    assert 0 <= stats["accepted_rows"] <= 2
    assert (tmp_path / "final_dataset.json").exists()

