"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]

    # Compute accuracy-based metrics (original behavior)
    target_tok = tok(" " + target_new["str"])["input_ids"]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_acc = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    # Compute probability-based metrics (for success/diff calculation)
    stuff_probs = test_batch_prediction_probs(
        model,
        tok,
        list(chain(*prob_prompts)),
        target_new["str"],
        target_true["str"],
    )

    # Predict for neighborhood prompts (dictionary format).
    neighborhood_acc = test_batch_prediction_acc(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    # Compute neighborhood probs
    neighborhood_probs = test_batch_prediction_probs(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        target_new["str"],
        target_true["str"],
    )

    acc_probs = stuff_acc + neighborhood_acc

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_acc = [acc_probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]

    # Structure the results as a dictionary - return BOTH acc and probs
    ret = {
        f"{key}_correct": ret_acc[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_acc

    # Add probability metrics for success/diff calculation
    ret["rewrite_prompts_probs"] = stuff_probs
    ret["paraphrase_prompts_probs"] = []
    ret["neighborhood_prompts_probs"] = neighborhood_probs

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()


def test_batch_prediction_probs(model, tok, prompts: typing.List[str], target_new: str, target_true: str):
    """
    Compute log probabilities for target_new and target_true given prompts.
    Returns list of dicts with 'target_new' and 'target_true' log probs.
    """
    new_tok = tok(f" {target_new}")["input_ids"]
    true_tok = tok(f" {target_true}")["input_ids"]
    max_len = max(len(new_tok), len(true_tok))

    # Pad tokens to same length
    new_tok = new_tok + [tok.pad_token_id] * (max_len - len(new_tok))
    true_tok = true_tok + [tok.pad_token_id] * (max_len - len(true_tok))

    prompt_tok = tok(
        [p for p in prompts for _ in range(2)],  # Duplicate for new and true
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    new_tok_t = torch.tensor([new_tok] * len(prompts)).to("cuda")
    true_tok_t = torch.tensor([true_tok] * len(prompts)).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        prompt_lens = prompt_tok["attention_mask"].sum(1).tolist()

    probs = []

    for i, p_len in enumerate(prompt_lens):
        new_prob = 0.0
        true_prob = 0.0

        # Compute log prob for each token in target
        for j in range(max_len):
            if new_tok[j] != tok.pad_token_id:
                log_prob = torch.nn.functional.log_softmax(
                    logits[i, p_len - 1 + j, :], dim=0
                )[new_tok[j]].item()
                new_prob += log_prob

            if true_tok[j] != tok.pad_token_id:
                log_prob = torch.nn.functional.log_softmax(
                    logits[i + len(prompts), p_len - 1 + j, :], dim=0
                )[true_tok[j]].item()
                true_prob += log_prob

        # Normalize by token count
        new_len = sum(1 for t in new_tok if t != tok.pad_token_id)
        true_len = sum(1 for t in true_tok if t != tok.pad_token_id)

        probs.append({
            "target_new": new_prob / new_len,
            "target_true": true_prob / true_len,
        })

    return probs
