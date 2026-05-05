"""
Sequential editing evaluation script.
Unlike evaluate.py, weights are NOT restored between edits.
Each case is applied one by one on top of the previously edited model.
Final evaluation is done once after all edits are applied.
"""
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from util.eval_utils.eval_utils_counterfact import compute_rewrite_quality_counterfact
from util.eval_utils.eval_utils_zsre import compute_rewrite_quality_zsre

from memit import MEMITHyperParams, apply_memit_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),  # 新增
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    model_path: str,  # 新增
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    cuda_device: int = None,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")
    # Instantiate vanilla model
    # determine device
    if cuda_device is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device)
        except Exception:
            # best-effort: ignore if setting device fails
            pass

    if type(model_name) is str:
        print("Instantiating model")
        if model_path is not None:
            # 本地路径模式（用于 LLaMA-3 等大模型）
            from util.hparams import get_model_path

            model_name_or_path = get_model_path(model_path, model_name)
        else:
            model_name_or_path = model_name
        # load model then move to chosen device (so we can catch OOM on transfer)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        try:
            model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(
                f"CUDA OOM or error moving model to {device} — falling back to CPU. Error: {e}"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, low_cpu_mem_usage=True
            )
            device = torch.device("cpu")
        tok = AutoTokenizer.from_pretrained(model_name_or_path)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # ------------------------------------------------------------------ #
    # Phase 1: Sequential editing — apply all edits WITHOUT weight reset  #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: Sequential editing (no weight restore) ===")
    all_records = list(ds)  # flatten dataset into a list
    total_exec_time = 0.0

    for i, record in enumerate(all_records):
        case_id = record["case_id"]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = (
            dict(cache_template=cache_template)
            if any(alg in alg_name for alg in ["ROME", "MEMIT"])
            else dict()
        )

        start = time()
        # apply_algo in-place (copy=False), discard weights_copy — no restore
        model, _ = apply_algo(
            model,
            tok,
            [{"case_id": case_id, **record["requested_rewrite"]}],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        total_exec_time += exec_time

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Edited {i + 1}/{len(all_records)} cases, "
                  f"last edit took {exec_time:.2f}s, "
                  f"total so far {total_exec_time:.2f}s")

    print(f"\nAll {len(all_records)} edits applied. "
          f"Total execution time: {total_exec_time:.2f}s "
          f"({total_exec_time / len(all_records):.2f}s/case)")

    # ------------------------------------------------------------------ #
    # Phase 2: Evaluate all cases on the fully-edited model              #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 2: Evaluating all cases on fully-edited model ===")
    eval_start = time()
    case_result_template = str(run_dir / "seq_{}_edits-case_{}.json")

    gen_test_vars = [snips, vec]
    for i, record in enumerate(all_records):
        case_id = record["case_id"]
        out_file = Path(case_result_template.format(len(all_records), case_id))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue

        metrics = {
            "case_id": case_id,
            "num_edits": len(all_records),
            "edit_index": i,          # position in the edit sequence
            "requested_rewrite": record["requested_rewrite"],
            "total_exec_time": total_exec_time,
            "post": ds_eval_method(
                model,
                tok,
                record,
                *(
                    gen_test_vars
                    if case_id % generation_test_interval == 0
                    else [None, None]
                ),
            ),
        }

        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

        if (i + 1) % 200 == 0:
            print(f"  Evaluated {i + 1}/{len(all_records)} cases ...")

    eval_time = time() - eval_start
    print(f"Evaluation done. Took {eval_time:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT",  "PMET"],# 新增 PMET
        default="MEMIT",
        required=True,
    )
    parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Local model path (e.g., /home/wentao/xzw/LLM/)",
)
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="zsre",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=None,
        help="GPU index to use (e.g., 4)",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.model_path,  # 新增
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        cuda_device=args.cuda_device,
        use_cache=args.use_cache,
    )
