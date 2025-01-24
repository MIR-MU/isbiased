import logging
import os
from typing import List, Optional, Union, Dict

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from isbiased.bias_significance import BiasSignificanceMeasure

bias_significance = BiasSignificanceMeasure()

logger = logging.getLogger()


def pick_dataset(task: str, dataset: str, datasets_root: str, split: str = "validation") -> Dataset:
    if task.lower() == "qa":
        if dataset == "squad":
            return load_dataset("squad")[split]
        elif dataset == "adversarial_qa":
            return load_dataset("adversarial_qa", "adversarialQA")[split]
        elif dataset == "nq":
            assert split == "validation"
            return Dataset.from_pandas(pd.read_json(os.path.join(datasets_root, 'nq_dev_formatted.json')))
        elif dataset == "triviaqa":
            assert split == "validation"
            return Dataset.from_pandas(pd.read_json(os.path.join(datasets_root, 'triviaqa_dev_formatted.json')))
        else:
            raise ValueError("Unknown dataset %s" % dataset)
    else:
        # TODO: add more datasets in SQuAD format!
        raise NotImplementedError("Unknown task %s" % task)


def eval_datasets(model_or_path: Union[str, PreTrainedModel],
                  task: str,
                  dataset_ids: List[str],
                  dataset_root: str,
                  firstn: Optional[int] = 0,
                  batch_size: Optional[int] = 2,
                  tokenizer: Optional[PreTrainedTokenizer] = None) -> Dict[str, float]:
    out_dict = {}
    for dataset_id in dataset_ids:
        print("Evaluating on dataset %s" % dataset_id)
        dataset = pick_dataset(task, dataset_id, dataset_root)
        dataset = dataset.select(range(firstn)) if firstn else dataset
        perf, pred_dataset = bias_significance.evaluate_model_on_dataset(model_or_path, dataset, batch_size, tokenizer)
        out_dict[dataset_id] = perf['exact_match']

    out_dict["datasets_avg"] = sum(out_dict.values()) / len(out_dict)

    return out_dict


def eval_shortcuts(model_or_path: Union[str, PreTrainedModel],
                   task: str,
                   dataset_id: str,
                   dataset_root: str,
                   shortcuts: List[str],
                   firstn: Optional[int] = 0,
                   batch_size: Optional[int] = 2,
                   tokenizer: Optional[PreTrainedTokenizer] = None) -> Dict[str, float]:

    dataset = pick_dataset(task, dataset_id, dataset_root)
    dataset = dataset.select(range(firstn)) if firstn else dataset

    perf, pred_dataset = bias_significance.evaluate_model_on_dataset(model_or_path, dataset, batch_size, tokenizer)
    logger.warning("Given model's ID performance: %s", perf)

    shortcuts_eval = {}

    for shortcut in shortcuts:
        threshold_distance_dictionary, _ = bias_significance.find_longest_distance(pred_dataset, shortcut)
        shortcuts_eval[shortcut] = threshold_distance_dictionary[1]

    shortcuts_all_vals = [s if s != -1 else 0 for s in shortcuts_eval.values()]
    shortcuts_eval["shortcuts_avg"] = sum(shortcuts_all_vals) / len(shortcuts_all_vals)

    return shortcuts_eval
