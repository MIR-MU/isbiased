import argparse
from typing import Tuple

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, \
    PreTrainedModel, Trainer

from learnedMixinH.transformers_impl.loss import LearnedMixinHLoss
from learnedMixinH.transformers_impl.model import LMixBertForQuestionAnswering
from learnedMixinH.transformers_impl.utils import prepare_train_features


def infer_model_start_end_logits(qa_model: PreTrainedModel,
                                 dataset: Dataset,
                                 batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qa_model = qa_model.to(device)

    with torch.no_grad():
        start_logits_l = []
        end_logits_l = []

        for batch_offset in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size,
                                 desc="Inferring logits of QA models"):
            batch_encodings = dataset[batch_offset: batch_offset + batch_size]
            outputs = qa_model(**{k: torch.tensor(v).to(device) for k, v in batch_encodings.items()})
            start_logits_l.append(outputs.start_logits)
            end_logits_l.append(outputs.end_logits)

    return torch.vstack(start_logits_l), torch.vstack(end_logits_l)


def create_bias_dataset(dataset: Dataset, bias: PreTrainedModel) -> Dataset:
    """
    Combine distillation examples (teacher_preds and bias_preds) into one dataset that can be fed through Trainer API
    :param dataset: Dataset in HF format
    :param teacher: Teacher QA model to distil from
    :param bias: Biased QA model to downscale the Teacher's prediction with
    :return Dataset
    """

    bias_start_logits, bias_end_logits = infer_model_start_end_logits(bias, dataset["train"])

    dataset['train'] = dataset['train'].add_column('bias_probs_start', bias_start_logits.tolist())
    dataset['train'] = dataset['train'].add_column('bias_probs_end', bias_end_logits.tolist())

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--trained_model", default="bert-base-cased", type=str,
                        help="Model to be debiased")
    parser.add_argument("--penalty", default=0.3, type=float,
                        help="Entropy regularization penalty (H).")
    parser.add_argument("--biased_model_path", default="bert-base-cased", type=str,
                        help="Pre-trained biased model")
    parser.add_argument("--dataset", default="squad", type=str,
                        help="HuggingFace dataset name, e.g.: 'squad'")
    parser.add_argument("--output_path",
                        default="./results",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum length of a feature (question and context). \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--doc_stride",
                        default=128,
                        type=int,
                        help="The authorized overlap between two parts of the context when splitting is needed. "
                             "Using max_seq_length with doc_stride ensures truncation and padding, "
                             "but keep the overflows using a stride. This results in in one example possible giving "
                             "several features when a context is long, with features having a overlaps "
                             "in theirs contexts.")

    parser.add_argument("--do_train",
                        default=False,
                        help="Whether to run training.")
    parser.add_argument("--train_firstn",
                        default=0,
                        type=int,
                        help="Subset of the training data. For testing.")
    parser.add_argument("--eval_firstn",
                        default=0,
                        type=int,
                        help="Subset of the training data. For testing.")

    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Batch for training that fits into memory.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size for evaluation that fits into memory.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    print("Training script started!")

    no_cuda = args.no_cuda
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

    print("Device:  ", device)

    model_checkpoint = args.trained_model
    print("Model:   ", model_checkpoint)
    print("Bias model:   ", args.biased_model_path)
    debiased_name = "debiased-lmix-" + model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    loss_fn = LearnedMixinHLoss(penalty=args.penalty)
    lmix_model = LMixBertForQuestionAnswering.from_pretrained(args.trained_model,
                                                              loss_fn=loss_fn,
                                                              output_hidden_states=True)
    lmix_model.to(device)

    # load teacher predictions and biased predictions
    # dataset = load_dataset(args.dataset)
    dataset = load_dataset(args.dataset)
    if args.train_firstn:
        dataset["train"] = dataset["train"].select(range(args.train_firstn))
    if args.eval_firstn:
        dataset["validation"] = dataset["validation"].select(range(args.eval_firstn))

    tokenized_squad = dataset.map(prepare_train_features, batched=True,
                                  remove_columns=dataset["train"].column_names,
                                  fn_kwargs={'tokenizer': tokenizer, 'args': args})
    print("Got dataset...")
    data_collator = DefaultDataCollator()

    bias_model = AutoModelForQuestionAnswering.from_pretrained(args.biased_model_path)

    lmix_dataset = create_bias_dataset(tokenized_squad, bias_model)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        evaluation_strategy="steps",
        eval_steps=1000,  # Evaluation and Save happens every 200 steps
        save_steps=1000,
        logging_steps=1000,
        save_total_limit=50,  # Only last 10 models are saved. Older ones are deleted.
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=lmix_model,
        args=training_args,
        train_dataset=lmix_dataset["train"],
        eval_dataset=lmix_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
    )

    trainer.train()

    trainer.save_model(debiased_name)


if __name__ == '__main__':
    main()
