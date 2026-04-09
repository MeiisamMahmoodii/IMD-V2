import json
from pathlib import Path

def main():
    root = Path("experiments/results")
    outputs = sorted(root.glob("exp_*.json"))
    print(json.dumps({"count": len(outputs), "files": [str(p) for p in outputs]}, indent=2))

if __name__ == "__main__":
    main()
