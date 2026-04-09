import json
from pathlib import Path

def main():
    root = Path("experiments/results")
    outputs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("exp_*.json"))]
    (root / "reports" / "final_findings_summary.md").write_text(f"# Final Findings\n\nExperiments: {len(outputs)}", encoding="utf-8")

if __name__ == "__main__":
    main()
