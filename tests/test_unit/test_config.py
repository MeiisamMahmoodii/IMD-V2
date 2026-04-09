from pathlib import Path
from isomorphic.config import ConfigManager

def test_yaml_roundtrip(tmp_path: Path):
    payload = {"a": 1, "b": {"c": True}}
    out = tmp_path / "c.yaml"
    ConfigManager.save_yaml(payload, out)
    assert ConfigManager.load_yaml(out) == payload
