from kvblock.bench.sparse_correctness_poc import main


if __name__ == "__main__":
    raise SystemExit(main(["--total-tokens", "256", "--keep-recent-blocks", "4"]))
