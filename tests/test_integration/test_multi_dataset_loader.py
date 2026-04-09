from isomorphic.loader import DatasetLoader

def test_load_all_datasets():
    for name in ["toxigen", "jigsaw", "hatexplain", "sbic", "ethos"]:
        ds = DatasetLoader.load(name, {"name": name, "limit": 5})
        assert len(ds.seeds) == 5
