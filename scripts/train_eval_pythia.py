import argparse
from typing import Optional, Union, Dict, Iterable, Iterator

import torch
import wandb
from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.objectives.CLM import DataCollatorForCausalLM
from adaptor.objectives.objective_base import SupervisedObjective, Objective
from adaptor.objectives.seq2seq import SequentialMixin
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy, Head
from transformers import BatchEncoding, PreTrainedModel, AutoModelForCausalLM

from scripts.utils import eval_datasets, eval_shortcuts, pick_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default="EleutherAI/pythia-14m", type=str)
parser.add_argument("--start_checkpoint", default=0, type=int)
parser.add_argument("--end_checkpoint", default=80000, type=int)
parser.add_argument("--checkpoint_step", default=20000, type=int)
parser.add_argument("--task", default="QA", type=str)
parser.add_argument("--shortcuts", help="Comma-separated list of shortcuts to evaluate. "
                                        "See default value for a list of options", required=True, type=str,
                    default="similar_words,distances,kth_sentence,cosine_similarity,answer_length,max_sim_ents,answer_subject_positions")
parser.add_argument("--id_dataset", help="Training dataset identifier", default="squad")
parser.add_argument("--ood_datasets", help="Comma-separated list of dataset identifiers. "
                                           "For QA, choose from `squad,nq,trivia_qa,adversarial_qa,news_qa,search_qa`. "
                                           "For NLI, choose from `mnli,anli,contract_nli,wanli`",
                    required=True, type=str)
parser.add_argument("--datasets_root", help="A path to jsons of OOD datasets in SQuAD format", default="scripts/ood_datasets")
parser.add_argument("--firstn", help="Number of first-n samples for each dataset to evaluate with", default=0, type=int)
parser.add_argument("--batch_size", help="Inference batch_size", default=4, type=int)
parser.add_argument("--train_batch_size", help="Effective training batch_size", default=32, type=int)

args = parser.parse_args()

assert args.train_batch_size % args.batch_size == 0, "--train_batch_size must be divisible by --batch_size"

INPUT_TEMPLATE = "%s %s Answer:"  # must be consistent with bias_significance!

train_dataset = pick_dataset(args.task, args.id_dataset, args.datasets_root, split="train")
val_dataset = pick_dataset(args.task, args.id_dataset, args.datasets_root, split="validation")

train_inputs = [INPUT_TEMPLATE % (x["context"], x["question"]) for x in train_dataset]
train_labels = [x["answers"]["text"][0] for x in train_dataset]

val_inputs = [INPUT_TEMPLATE % (x["context"], x["question"]) for x in val_dataset]
val_labels = [x["answers"]["text"][0] for x in val_dataset]

training_arguments = AdaptationArguments(output_dir="checkpoints-txt2sql",
                                         learning_rate=5e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_patience=3,
                                         save_total_limit=4,
                                         do_train=True,
                                         do_eval=True,
                                         bf16=True,
                                         warmup_steps=100,
                                         gradient_accumulation_steps=args.train_batch_size // args.batch_size,
                                         logging_steps=10,
                                         eval_steps=2000,
                                         save_steps=2000,
                                         num_train_epochs=10,
                                         evaluation_strategy="steps",
                                         )


class CausalSequence2Sequence(SequentialMixin, SupervisedObjective):

    compatible_head: Head = Head.CLM

    def __init__(self, *args, mask_prompt_from_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prompt_from_loss = mask_prompt_from_loss
        self.collator = DataCollatorForCausalLM(self.tokenizer, self.compatible_head_model)

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Computes sequence2sequence loss
        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw outputs of language modeling head model
        :param labels: Token ids of expected outputs.
        :return: Single value of the loss, with grad_fn.
        """
        # note that currently we do not ignore padding from the loss, which might be desirable
        # - we have seen this to eliminate repetitive generations at some cases
        loss_fct = torch.nn.CrossEntropyLoss()
        # vocab-agnostic loss circumvents incorrectly-set vocab_size of some models (e.g. mt5)
        lm_loss = loss_fct(lm_logit_outputs.flatten(end_dim=1), labels.flatten())
        return lm_loss

    def _get_seq2seq_collated_iterator(self,
                                       source_texts: Iterable[str],
                                       target_texts: Iterable[str]) -> Iterator[BatchEncoding]:
        """
        Creates an iterator over batches of encoded `source_texts` as inputs and `target_texts` as labels.
        Override this to implement custom mapping, or unsupervised seq2seq objective. See e.g. BackTranslation.
        :param source_texts: Input texts.
        :param target_texts: Output (expected) texts to translate input texts into.
        :return: Iterator of encoded batches.
        """
        features_batch = []
        for source_text, target_text in zip(source_texts, target_texts):
            self.tokenizer.src_lang = self.source_lang_id
            self.tokenizer.tgt_lang = self.target_lang_id
            sample_features = self.tokenizer(source_text, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                sample_targets = self.tokenizer(target_text, truncation=True)
            if self.mask_prompt_from_loss:
                labels = ([-100] * len(sample_features.input_ids)) + sample_targets.input_ids
            else:
                labels = sample_features.input_ids + sample_targets.input_ids

            features_batch.append({"input_ids": sample_features.input_ids + sample_targets.input_ids,
                                   "attention_mask": sample_features.attention_mask + sample_targets.attention_mask,
                                   "labels": labels})
            if len(features_batch) == self.batch_size:
                yield self.collator(features_batch)
                features_batch = []

        if features_batch:
            # yield last nonempty residual batch
            yield self.collator(features_batch)


def compute_weights_diff(orig_model: PreTrainedModel, new_model: PreTrainedModel) -> torch.Tensor:
    orig_params = list(orig_model.parameters())
    new_params = list(new_model.parameters())
    assert len(orig_params) == len(new_params)
    # implicit assertion of matching dimensions in the operation (subtraction)

    all_diffs = []
    for orig_param, new_param in zip(orig_params, new_params):
        all_diffs.append((orig_param - new_param).abs().flatten())

    return torch.hstack(all_diffs)


orig_model = AutoModelForCausalLM.from_pretrained(args.base_model)


for checkpoint_step in range(args.start_checkpoint, args.end_checkpoint, args.checkpoint_step):
    lang_module = LangModule(args.base_model)

    seq_qa = CausalSequence2Sequence(lang_module,
                                     objective_args_for_head_config={"revision": "step%s" % checkpoint_step},
                                     texts_or_path=train_inputs,
                                     labels_or_path=train_labels,
                                     val_texts_or_path=val_inputs,
                                     val_labels_or_path=val_labels,
                                     batch_size=args.batch_size,
                                     objective_id="SQuAD")

    # Add pad token to all models if using pythia
    if seq_qa.tokenizer.pad_token is None and seq_qa.tokenizer.pad_token_id is None:
        seq_qa.compatible_head_model.pad_token = "<|endoftext|>"
        seq_qa.tokenizer.pad_token = "<|endoftext|>"

    schedule = ParallelSchedule(objectives=[seq_qa], args=training_arguments)

    adapter = Adapter(lang_module, schedule, args=training_arguments)

    run_config = {"num_parameters": seq_qa.compatible_head_model.num_parameters(),
                  "num_pretraining_steps": checkpoint_step}

    with wandb.init(project="pretraining-robustness", entity="transformersclub", config=run_config) as run:
        print("Started training %s-%s" % (args.base_model, "step%s" % checkpoint_step))
        adapter.train()
        print("Done training %s-%s" % (args.base_model, "step%s" % checkpoint_step))
        print("Creating logs")

        # after the training, log the resulting metrics
        trained_model = seq_qa.compatible_head_model

        wandb.log(run_config, step=adapter.state.global_step)

        ood_evaluations = eval_datasets(trained_model, args.task, args.ood_datasets.split(","), args.datasets_root,
                                        args.firstn, args.batch_size, lang_module.tokenizer)

        wandb.log(ood_evaluations, step=adapter.state.global_step)

        shortcuts_eval = eval_shortcuts(trained_model, args.task, args.id_dataset, args.datasets_root,
                                        args.shortcuts.split(","), args.firstn, args.batch_size, lang_module.tokenizer)

        wandb.log(shortcuts_eval, step=adapter.state.global_step)
        with torch.no_grad():
            weights_diff = compute_weights_diff(orig_model, trained_model)
        opt_logs = {"num_steps": adapter.state.global_step, "weights_diff": wandb.Histogram(weights_diff, num_bins=512)}

        wandb.log(opt_logs, step=adapter.state.global_step)
        print("Done creating logs")
