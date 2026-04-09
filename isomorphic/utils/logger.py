import csv
import json
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class ResultsLogger:
    def __init__(self, root: str | Path = "experiments/results"):
        self.root = Path(root)
        self.reports = self.root / "reports"
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "metadata.json"
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"experiments": []}, indent=2), encoding="utf-8")

    def log_experiment(self, experiment_id: str, payload: dict[str, Any]) -> Path:
        out = self.root / f"{experiment_id}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
        reg.setdefault("experiments", []).append({"experiment_id": experiment_id, "file": str(out)})
        self.registry_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        return out

    def write_csv(self, filename: str, rows: list[dict[str, Any]]) -> Path:
        out = self.reports / filename
        if not rows:
            out.write_text("", encoding="utf-8")
            return out
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return out

    def write_markdown(self, filename: str, content: str) -> Path:
        out = self.reports / filename
        out.write_text(content, encoding="utf-8")
        return out
