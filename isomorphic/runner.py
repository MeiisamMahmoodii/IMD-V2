import argparse
from isomorphic.pipeline import IsomorphicPipeline

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-model", default="huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2")
    p.add_argument("--samples", type=int, default=100)
    a = p.parse_args()
    pipe = IsomorphicPipeline(a.embedding_model)
    src = [f"source sentence {i}" for i in range(a.samples)]
    tgt = [f"target sentence {i}" for i in range(a.samples)]
    print(pipe.run_alignment(src, tgt, "exp_demo"))

if __name__ == "__main__":
    main()
