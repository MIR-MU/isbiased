import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer

import torch

from debiasing_methods.confidenceRegularization.overrides.loss import SmoothedDistillLoss
from debiasing_methods.confidenceRegularization.overrides.models import BertDistill, BertDistillForQuestionAnswering
from debiasing_methods.confidenceRegularization.utils import prepare_train_features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", default="bert-base-uncased", type=str,
                        help="Model to be debiased")

    parser.add_argument("--output_dir",
                        default="./results",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--preds_dir",
                        default="./dataset",
                        type=str,
                        help="Directory to save teacher predictions to.")
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
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
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

    model_checkpoint = args.model
    print("Model:   ", model_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    loss_fn = SmoothedDistillLoss()
    distilled_model = BertDistillForQuestionAnswering.from_pretrained(model_checkpoint, loss_fn=loss_fn)
    distilled_model.to(device)


    # load teacher predictions and biased predictions
    dataset = load_dataset("squad")
    tokenized_squad = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names,
                                  fn_kwargs={'tokenizer': tokenizer, 'args':args})
    print("Got dataset...")
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # TODO - not working now!
    # need to override Trainer, because of distillation
    trainer = Trainer(
        model=distilled_model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    # generate teacher probs here and put with dataset

    trainer.evaluate(tokenized_squad['validation'])
    debiased_name = "debiased-conf_reg-" + model_checkpoint
    trainer.save_model()


if __name__ == '__main__':
    main()


