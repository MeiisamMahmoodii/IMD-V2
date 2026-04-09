from pathlib import Path

def main():
    p = Path("experiments/results/plots")
    p.mkdir(parents=True, exist_ok=True)
    for name in ["anchor_comparison.pdf", "scaling_curves.pdf", "model_pair_comparison.pdf", "stability_curves.pdf", "semantic_histogram.pdf"]:
        (p / name).write_bytes(b"%PDF-1.4\n% scaffold plot\n")

if __name__ == "__main__":
    main()
