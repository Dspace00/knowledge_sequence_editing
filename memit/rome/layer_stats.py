import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float16", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        if ds_name == "wikipedia":
            import pyarrow as pa
            import pandas as pd
            from datasets import Dataset

            # 尝试多个可能的 Wikipedia Arrow 缓存路径（优先级从高到低）
            possible_paths = [
                # 21号服务器 data0 NVMe 盘（优先，上传的 wiki 语料）
                "/data0/xiezhiwei/wiki/wikipedia_train.arrow",
                # 138号服务器路径
                "/home/wentao/.cache/huggingface/datasets/wikipedia/20200501.en/1.0.0/009f923d9b6dd00c00c8cdc7f408f2b47f45dd4f5fb7982a21f9448f4afbe475/wikipedia-train.arrow",
                # 21号服务器 .cache 路径（如有）
                "/home/xiezhiwei/.cache/huggingface/datasets/wikipedia/20200501.en/1.0.0/009f923d9b6dd00c00c8cdc7f408f2b47f45dd4f5fb7982a21f9448f4afbe475/wikipedia-train.arrow",
            ]
            arrow_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    try:
                        print(f"Trying to open Wikipedia from: {p}")
                        with pa.memory_map(p, "r") as mmap:
                            reader = pa.ipc.open_stream(mmap)
                            table = reader.read_all()
                        df = table.to_pandas()
                        raw_ds = Dataset.from_pandas(df)
                        arrow_path = p
                        print(f"Successfully loaded Wikipedia from: {arrow_path}")
                        break
                    except Exception as e:
                        print(f"Failed to open {p}: {e}, trying next path...")
            else:
                print("Local Arrow file not found, using load_dataset() to load/download wikipedia...")
                raw_ds = load_dataset("wikipedia", "20200501.en")

        # wikitext 分支走默认逻辑（和原代码一致）
        if ds_name == "wikitext":
            raw_ds = load_dataset("wikitext", "wikitext-103-raw-v1")

        # 计算 maxlen
        npos = getattr(model.config, 'n_positions', None) or getattr(model.config, 'max_position_embeddings', 2048)
        if batch_tokens is not None and batch_tokens < npos:
            maxlen = batch_tokens
        else:
            maxlen = npos

        # 兼容新旧 datasets 版本：新版直接返回 Dataset，旧版返回 DatasetDict
        if isinstance(raw_ds, DatasetDict):
            raw_ds = raw_ds["train"]
        return TokenizedDataset(raw_ds, tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 1  # 8B 模型 + RTX 4090 D (24GB)，batch_size=1 避免 OOM
    npos = getattr(model.config, 'n_positions', None) or model.config.max_position_embeddings
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = f"_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    if not filename.exists() and download:
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,  # 避免 CPU 预取占用额外内存
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                with torch.no_grad():
                    # forward 在 GPU 上做
                    batch = dict_to_(batch, "cuda")
                    with Trace(
                        model, layer_name, retain_input=True, retain_output=False, stop=True
                    ) as tr:
                        model(**batch)
                    batch = dict_to_(batch, "cpu")  # 立即释放 GPU batch 显存
                    # tr.input 也在 GPU 上，移到 CPU 再处理
                    feats = flatten_masked_batch(tr.input.detach().cpu(), batch["attention_mask"])
                    # feats = flatten_masked_batch(tr.output.detach().cpu(), batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)
                    torch.cuda.empty_cache()  # 每 batch 后显式释放 GPU 缓存
    return stat


if __name__ == "__main__":
    main()
