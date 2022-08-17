import argparse
import os

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer, \
    EarlyStoppingCallback

import torch

import pandas as pd

from debiasing_methods.confidenceRegularization.utils import prepare_train_features, get_model_filename, \
    get_preds_filename, get_dataset_path

from isbiased.bias_significance import BiasSignificanceMeasure

dirname = os.getcwd()


def is_saved(teacher_model: str, args):
    for name in ['easy', 'hard']:
        filename = "_".join([args.dataset, os.path.basename(teacher_model), args.bias, name])
        path = os.path.join(dirname, "dataset", filename)
        if not os.path.isdir(path):
            return False
    return True


def save_for_later(dataset, teacher_model: str, is_biased: str, args):
    is_biased_name = 'easy' if is_biased else 'hard'
    filename = "_".join([args.dataset, os.path.basename(teacher_model), args.bias, is_biased_name])
    path = os.path.join(dirname, "dataset", filename)
    dataset.save_to_disk(path)


def load_saved_split(teacher_model: str, is_biased: bool, args):
    is_biased_name = 'easy' if is_biased else 'hard'
    filename = "_".join([args.dataset, os.path.basename(teacher_model), args.bias, is_biased_name])
    path = os.path.join(dirname, "dataset", filename)
    return load_from_disk(path)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", default="bert-base-uncased", type=str,
                        help="Model to be trained on biased samples")
    parser.add_argument("--teacher_model", default="./saved_models/bert-base-uncased_finetuned_baseline", type=str,
                        help="Pre-trained teacher model to extract bias measurements from.")
    parser.add_argument("--bias", default="distances", type=str,
                        help="On which bias to train model. Supports all biases of 'isbiased' lib. "
                             "Possible values: 'similar_words','distances','kth_sentence','cosine_similarity',"
                             "'answer_length','max_sim_ents','answer_subject_positions'")
    parser.add_argument("--dataset", default="squad", type=str,
                        help="HuggingFace dataset name, e.g.: 'squad'")
    parser.add_argument("--output_dir",
                        default="./saved_models",
                        type=str,
                        help="The output directory where the model and checkpoints will be written.")
    # parser.add_argument("--preds_dir",
    #                     default="./dataset",
    #                     type=str,
    #                     help="Directory to save model predictions to.")
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
                        default=True,
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

    model = args.model
    # biased_checkpoint = model + "-biased-" + args.bias
    biased_checkpoint = get_model_filename(model,args.bias,args.dataset, True)
    print("Model:   ", model)
    model_save_path = os.path.join(dirname, 'saved_models', biased_checkpoint)

    teacher_model = args.teacher_model
    print("Teacher model:   ", teacher_model)

    biased_tokenizer = AutoTokenizer.from_pretrained(model)
    biased_model = AutoModelForQuestionAnswering.from_pretrained(model)
    biased_model.to(device)

    squad_en = load_dataset(args.dataset)

    if not is_saved(teacher_model, args):
        measurer = BiasSignificanceMeasure(squad_en['train'])
        measurer.evaluate_model_on_dataset(teacher_model, squad_en['train'])
        measurer.compute_heuristic(args.bias)
        biasedDataset, unbiasedDataset = measurer.split_data_by_heuristics(squad_en['train'], args.bias)

        save_for_later(biasedDataset, teacher_model, True, args)
        save_for_later(unbiasedDataset, teacher_model, False, args)

    biasedDataset = load_saved_split(teacher_model, True, args)
    unbiasedDataset = load_saved_split(teacher_model, False, args)

    biased_dataset_train = biasedDataset.map(prepare_train_features, batched=True,
                                      remove_columns=biasedDataset.column_names,
                                      fn_kwargs={'tokenizer': biased_tokenizer, 'args': args})

    unbiased_dataset_val = biasedDataset.select(range(1000)).map(prepare_train_features, batched=True,
                                                        remove_columns=unbiasedDataset.column_names,
                                                        fn_kwargs={'tokenizer': biased_tokenizer, 'args': args})

    tokenized_dataset_train = squad_en.map(prepare_train_features, batched=True,
                                      remove_columns=squad_en['train'].column_names,
                                      fn_kwargs={'tokenizer': biased_tokenizer, 'args': args})

    print("Got dataset...")
    data_collator = DefaultDataCollator()

    if args.do_train:
        training_args = TrainingArguments(
            output_dir=model_save_path,
            evaluation_strategy="steps",
            eval_steps=200,  # Evaluation and Save happens every 200 steps
            save_steps=200,
            logging_steps=200,
            save_total_limit=5,  # Only last 5 models are saved. Older ones are deleted.
            report_to="none",
            # evaluation_strategy="no",  # for testing
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.train_batch_size,
            max_steps=2,  # for testing
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_proportion,
            weight_decay=0.01,
            load_best_model_at_end=True,
            disable_tqdm=False
        )

        trainer = Trainer(
            model=biased_model,
            args=training_args,
            train_dataset=biased_dataset_train,
            eval_dataset=unbiased_dataset_val,
            tokenizer=biased_tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model(model_save_path)

        print(f"Model trained 🤗 state_dict saved at:    {model_save_path}")

    training_args = TrainingArguments(
        output_dir=model_save_path,
        # evaluation_strategy="no",  # for testing
        per_device_eval_batch_size=args.eval_batch_size,
        # max_steps=2,  # for testing
        disable_tqdm=False
    )

    trainer = Trainer(
        model=biased_model,
        args=training_args,
        tokenizer=biased_tokenizer,
        data_collator=data_collator,
    )

    # prediction of whole dataset - predictions of BIASED model (trained on biased examples)
    predictions, label_ids, metrics = trainer.predict(test_dataset=tokenized_dataset_train['train'].select(range(100)))
    # predictions contain outputs of net for all examples shape:
    #       2(as for start_logits and end_logits) * num_examples(dataset size) * num_outputs(output dimension of net)
    #           first row of shape is start_logits, second row is end_logits

    data = pd.DataFrame()
    data['start_logits'] = pd.Series(predictions[0].tolist())
    data['end_logits'] = pd.Series(predictions[1].tolist())
    # predictions_path = os.path.join(args.preds_dir, "biased_preds" + "_" + biased_checkpoint + "_" + args.dataset +".json")
    predictions_path = get_dataset_path(get_preds_filename(args.model, args.bias, args.dataset,True))
    data.to_json(predictions_path)

    print(f"Knowledge distillation completed! 👌 \n"
          f"Biased predictions saved at:   {predictions_path}")


if __name__ == '__main__':
    main()
