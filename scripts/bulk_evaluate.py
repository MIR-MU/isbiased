"""
Running example:
python scripts/bulk_evaluate.py \
 --task QA \
 --base_models models/t5-large-squad,deepset/roberta-base-squad2-distilled \
 --shortcuts similar_words,distances \
 --ood_datasets triviaqa,nq,adversarial_qa \
 --datasets_root isbiased/datasets
 --firstn 200
"""

import argparse
import os
from datetime import datetime

import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm

from isbiased.bias_significance import BiasSignificanceMeasure
from scripts.utils import pick_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--task", help="Task identifier. One of `QA`, `NLI.`", default="QA", type=str)
parser.add_argument("--models", help="Comma-separated list of models to evaluate", required=True, type=str)
# parser.add_argument("--id_dataset", help="In-distribution dataset used to measure bias significance.", required=True)
parser.add_argument("--shortcuts", help="Comma-separated list of shortcuts to evaluate. "
                                        "See default value for a list of options", required=True, type=str,
                    default="similar_words,distances,kth_sentence,cosine_similarity,answer_length,max_sim_ents,answer_subject_positions")
parser.add_argument("--ood_datasets", help="Comma-separated list of dataset identifiers. "
                                           "For QA, choose from `squad,nq,trivia_qa,adversarial_qa,news_qa,search_qa`. "
                                           "For NLI, choose from `mnli,anli,contract_nli,wanli`",
                    required=True, type=str, default="")
parser.add_argument("--datasets_root", help="A path to jsons of OOD datasets in SQuAD format", default="scripts/ood_datasets")
parser.add_argument("--firstn", help="Number of first-n samples for each dataset to evaluate with", default=0, type=int)
args = parser.parse_args()

OUTPUT_JSONL = {"model": None, "dateset": None, "measure_type": None, "value": None,
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M")}


def print_output_jsonl(model: str, dateset: str, measure_type: str, metric: str, value: float) -> None:
    output_json = {"model": model, "dateset": dateset, "measure_type": measure_type, "metric": metric, "value": value,
                   "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M")}
    print(output_json)


if args.task.lower() == "nli":
    raise NotImplementedError("NLI task not yet implemented")

for model_id in args.models.split(","):
    if args.task.lower() == "qa":
        id_dataset_id = "squad"
        id_dataset = pick_dataset(args.task, id_dataset_id, args.datasets_root)
    else:
        raise NotImplementedError("NLI task not yet implemented")

    bias_significance = BiasSignificanceMeasure()
    eval_dataset = id_dataset.select(range(args.firstn)) if args.firstn else id_dataset
    id_performance, pred_dataset = bias_significance.evaluate_model_on_dataset(model_id, eval_dataset)
    [print_output_jsonl(model_id, id_dataset_id, "perf", metric, val)
     for metric, val in id_performance.items()]
    for shortcut in tqdm(args.shortcuts.split(","), desc="Evaluating shortcuts reliance"):
        if not shortcut:
            continue
        threshold_distance_dictionary, dataset = bias_significance.find_longest_distance(pred_dataset, shortcut)
        # TODO: currently, we do not support f1-score as metric for bias distance eval
        distance_per_metric = {"exact_match": threshold_distance_dictionary[1]}
        [print_output_jsonl(model_id, id_dataset_id, "shortcut-%s" % shortcut, metric, val)
         for metric, val in distance_per_metric.items()]

    for ood_dataset_id in args.ood_datasets.split(","):
        if not ood_dataset_id:
            continue
        ood_dataset = pick_dataset(args.task, ood_dataset_id, args.datasets_root)
        ood_dataset = ood_dataset.select(range(args.firstn)) if args.firstn else ood_dataset
        ood_performance, pred_dataset = bias_significance.evaluate_model_on_dataset(model_id, ood_dataset)
        [print_output_jsonl(model_id, ood_dataset_id, "perf", metric, val)
         for metric, val in ood_performance.items()]

# TODO: set up parameters to test locally, see what happens
# TODO: parameter for first-n samples from each dataset (to test locally)

print("done")
