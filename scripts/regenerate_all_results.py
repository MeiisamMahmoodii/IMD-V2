from scripts.setup_datasets import main as setup
from scripts.run_experiments import main as run_experiments
from scripts.compute_statistics import main as stats

def main():
    setup()
    run_experiments()
    stats()

if __name__ == "__main__":
    main()
