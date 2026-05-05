"""
Summarize results from sequential editing runs.
Reads results from results/<dir_name>/run_XXX/seq_*.json
"""
import json
import statistics
from pathlib import Path
from argparse import ArgumentParser

from util.globals import RESULTS_DIR


def load_records(run_dir, prefix="seq_"):
    records = []
    for f in sorted(run_dir.glob(f"{prefix}*.json")):
        with open(f) as fp:
            records.append(json.load(fp))
    return records


def accuracy(values):
    """Compute mean accuracy from a list of per-token correctness lists."""
    flat = [v for vals in values for v in vals]
    return sum(flat) / len(flat) if flat else 0.0


def main(dir_name: str, run: str):
    run_dir = RESULTS_DIR / dir_name / run
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    records = load_records(run_dir)
    if not records:
        print("No sequential result files found.")
        return

    num_edits = records[0]["num_edits"]
    total_exec_time = records[0]["total_exec_time"]

    # Collect per-case metrics
    rewrite_accs = []
    paraphrase_accs = []
    neighborhood_accs = []

    for r in records:
        post = r["post"]
        rewrite_accs.append(post["rewrite_prompts_correct"])
        paraphrase_accs.append(post["paraphrase_prompts_correct"])
        neighborhood_accs.append(post["neighborhood_prompts_correct"])

    # Average across all cases
    rewrite_mean = accuracy(rewrite_accs)
    paraphrase_mean = accuracy(paraphrase_accs)
    neighborhood_mean = accuracy(neighborhood_accs)

    # Std across cases
    rewrite_stds = [statistics.stdev(vals) for vals in rewrite_accs if len(vals) > 1]
    paraphrase_stds = [statistics.stdev(vals) for vals in paraphrase_accs if len(vals) > 1]
    neighborhood_stds = [statistics.stdev(vals) for vals in neighborhood_accs if len(vals) > 1]

    rewrite_std = statistics.mean(rewrite_stds) if rewrite_stds else 0.0
    paraphrase_std = statistics.mean(paraphrase_stds) if paraphrase_stds else 0.0
    neighborhood_std = statistics.mean(neighborhood_stds) if neighborhood_stds else 0.0

    # Per-edit-index analysis: how does rewrite_acc change as more edits accumulate?
    # Sort by edit_index
    by_index = sorted(records, key=lambda r: r.get("edit_index", 0))
    early = by_index[: len(by_index) // 3]
    middle = by_index[len(by_index) // 3: 2 * len(by_index) // 3]
    late = by_index[2 * len(by_index) // 3:]

    def segment_mean(seg):
        return accuracy([r["post"]["rewrite_prompts_correct"] for r in seg])

    early_mean = segment_mean(early)
    middle_mean = segment_mean(middle)
    late_mean = segment_mean(late)

    print(f"\n{'='*60}")
    print(f"Sequential Editing Summary — {dir_name}/{run}")
    print(f"{'='*60}")
    print(f"Num cases edited : {num_edits}")
    print(f"Total exec time  : {total_exec_time:.2f}s "
          f"({total_exec_time / num_edits:.4f}s/case)")
    print()
    print(f"Overall Metrics (evaluated on fully-edited model):")
    print(f"  post_rewrite_acc     : {rewrite_mean*100:.2f}% ± {rewrite_std*100:.2f}")
    print(f"  post_paraphrase_acc  : {paraphrase_mean*100:.2f}% ± {paraphrase_std*100:.2f}")
    print(f"  post_neighborhood_acc : {neighborhood_mean*100:.2f}% ± {neighborhood_std*100:.2f}")
    print()
    print(f"Rewrite Acc by Edit Order (forgetting signal):")
    print(f"  Early edits (first 1/3)  : {early_mean*100:.2f}%")
    print(f"  Middle edits (middle 1/3): {middle_mean*100:.2f}%")
    print(f"  Late edits (last 1/3)   : {late_mean*100:.2f}%")
    print()
    print(f"Results dir: {run_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_name", default="MEMIT")
    parser.add_argument("--run", default=None, help="e.g. run_000")
    args = parser.parse_args()

    # Auto-detect latest run if not specified
    if args.run is None:
        alg_dir = RESULTS_DIR / args.dir_name
        run_dirs = sorted(
            [d for d in alg_dir.iterdir() if d.name.startswith("run_")],
            key=lambda d: int(d.name.split("_")[-1]),
        )
        if run_dirs:
            args.run = run_dirs[-1].name
            print(f"Auto-detected latest run: {args.run}")
        else:
            print("No run directories found.")
            exit(1)

    main(args.dir_name, args.run)
