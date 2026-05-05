"""
Summarize results from sequential editing runs.
Reads results from results/<dir_name>/run_XXX/seq_*.json
"""
import json
import statistics
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from scipy.stats import hmean

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


def compute_success(probs_list):
    """
    Compute success rate from a list of probs dicts.
    For rewrite/paraphrase: success = target_true > target_new (new should be more likely)
    Returns ratio of cases where new is more probable than true.
    """
    if not probs_list or all(not p for p in probs_list):
        return None
    count = 0
    total = 0
    for p in probs_list:
        if p and "target_new" in p and "target_true" in p:
            if p["target_true"] > p["target_new"]:
                count += 1
            total += 1
    return count / total if total > 0 else None


def compute_diff(probs_list):
    """
    Compute diff metric from a list of probs dicts.
    diff = exp(-target_new) - exp(-target_true)
    Returns mean of diff values.
    """
    if not probs_list or all(not p for p in probs_list):
        return None
    diffs = []
    for p in probs_list:
        if p and "target_new" in p and "target_true" in p:
            diff = np.exp(-p["target_new"]) - np.exp(-p["target_true"])
            diffs.append(diff)
    return np.mean(diffs) if diffs else None


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

    # For success and diff metrics
    rewrite_success = []
    paraphrase_success = []
    neighborhood_success = []

    rewrite_diff = []
    paraphrase_diff = []
    neighborhood_diff = []

    for r in records:
        post = r["post"]
        rewrite_accs.append(post["rewrite_prompts_correct"])
        paraphrase_accs.append(post["paraphrase_prompts_correct"])
        neighborhood_accs.append(post["neighborhood_prompts_correct"])

        # Compute success and diff from probs
        if "rewrite_prompts_probs" in post:
            rewrite_success.append(compute_success(post["rewrite_prompts_probs"]))
            rewrite_diff.append(compute_diff(post["rewrite_prompts_probs"]))

        if "paraphrase_prompts_probs" in post:
            paraphrase_success.append(compute_success(post["paraphrase_prompts_probs"]))
            paraphrase_diff.append(compute_diff(post["paraphrase_prompts_probs"]))

        if "neighborhood_prompts_probs" in post:
            neighborhood_success.append(compute_success(post["neighborhood_prompts_probs"]))
            neighborhood_diff.append(compute_diff(post["neighborhood_prompts_probs"]))

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

    # Success metrics
    rewrite_success_rate = np.nanmean([s for s in rewrite_success if s is not None]) if rewrite_success else None
    paraphrase_success_rate = np.nanmean([s for s in paraphrase_success if s is not None]) if paraphrase_success else None
    neighborhood_success_rate = np.nanmean([s for s in neighborhood_success if s is not None]) if neighborhood_success else None

    # Diff metrics
    rewrite_diff_mean = np.nanmean([d for d in rewrite_diff if d is not None]) if rewrite_diff else None
    paraphrase_diff_mean = np.nanmean([d for d in paraphrase_diff if d is not None]) if paraphrase_diff else None
    neighborhood_diff_mean = np.nanmean([d for d in neighborhood_diff if d is not None]) if neighborhood_diff else None

    # Score (hmean) based on success
    if all(x is not None for x in [rewrite_success_rate, paraphrase_success_rate, neighborhood_success_rate]):
        score_success = hmean([rewrite_success_rate, paraphrase_success_rate, neighborhood_success_rate])
    else:
        score_success = None

    # Score (hmean) based on acc
    score_acc = hmean([rewrite_mean, paraphrase_mean, neighborhood_mean])

    # Per-edit-index analysis: how does rewrite_acc change as more edits accumulate?
    # Sort by edit_index
    by_index = sorted(records, key=lambda r: r.get("edit_index", 0))
    early = by_index[: len(by_index) // 3]
    middle = by_index[len(by_index) // 3: 2 * len(by_index) // 3]
    late = by_index[2 * len(by_index) // 3:]

    def segment_mean_acc(seg):
        return accuracy([r["post"]["rewrite_prompts_correct"] for r in seg])

    early_mean = segment_mean_acc(early)
    middle_mean = segment_mean_acc(middle)
    late_mean = segment_mean_acc(late)

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
    print(f"  post_score_acc       : {score_acc*100:.2f}%")
    print()
    if rewrite_success_rate is not None:
        print(f"  post_rewrite_success     : {rewrite_success_rate*100:.2f}%")
        print(f"  post_paraphrase_success  : {paraphrase_success_rate*100:.2f}%")
        print(f"  post_neighborhood_success : {neighborhood_success_rate*100:.2f}%")
        print(f"  post_score_success       : {score_success*100:.2f}%")
        print()
    if rewrite_diff_mean is not None:
        print(f"  post_rewrite_diff     : {rewrite_diff_mean*100:.2f}")
        print(f"  post_paraphrase_diff  : {paraphrase_diff_mean*100:.2f}")
        print(f"  post_neighborhood_diff : {neighborhood_diff_mean*100:.2f}")
        print()
    print(f"Rewrite Acc by Edit Order (forgetting signal):")
    print(f"  Early edits (first 1/3)  : {early_mean*100:.2f}%")
    print(f"  Middle edits (middle 1/3): {middle_mean*100:.2f}%")
    print(f"  Late edits (last 1/3)   : {late_mean*100:.2f}%")
    print()
    print(f"Results dir: {run_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_name", default="PMET")
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
