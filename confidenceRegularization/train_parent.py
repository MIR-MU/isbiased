import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer

import torch

import pandas as pd

from confidenceRegularization.utils import prepare_train_features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", default="./saved_models/bert-base-uncased_finetuned_baseline", type=str,
                        help="Bert pre-trained model")
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
                        action='store_true',
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
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)

    dataset = load_dataset("squad")
    tokenized_squad = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names,
                                  fn_kwargs={'tokenizer': tokenizer})
    print("Got dataset...")
    data_collator = DefaultDataCollator()

    if args.do_train:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            # evaluation_strategy="epoch",
            evaluation_strategy="no",  # for testing
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradien_accumulation_steps,
            per_device_eval_batch_size=args.train_batch_size,
            max_steps=2,  # for testing
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_proportion,
            weight_decay=0.01,
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_squad["train"],
            eval_dataset=tokenized_squad["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model(model_checkpoint)

        print(f"Model trained 🤗 state_dict saved at:    {model_checkpoint}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # evaluation_strategy="epoch",
        evaluation_strategy="no",  # for testing
        per_device_eval_batch_size=args.eval_batch_size,
        max_steps=2,  # for testing
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # predictions, label_ids, metrics = trainer.predict(test_dataset=tokenized_squad['train'].train_test_split(train_size=5)['train'])
    predictions, label_ids, metrics = trainer.predict(test_dataset=tokenized_squad['train'])
    # predictions contain outputs of net for all examples shape:
    #       2(as for start_logits and end_logits) * num_examples(dataset size) * num_outputs(output dimension of net)
    #           first row of shape is start_logits, second row is end_logits

    data = pd.DataFrame()
    data['start_logits'] = pd.Series(predictions[0].tolist())
    data['end_logits'] = pd.Series(predictions[1].tolist())
    predictions_path = os.path.join(args.preds_dir,"teacher_preds" + "_" + os.path.basename(model))
    data.to_json(predictions_path)

    print(f"Knowledge distillation completed! 👌 \n"
          f"Teacher predictions saved at:   {predictions_path}")


if __name__ == '__main__':
    main()
