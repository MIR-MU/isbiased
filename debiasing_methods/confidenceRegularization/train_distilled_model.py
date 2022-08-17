import argparse
import os

import pandas as pd
from pandas import DataFrame

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer

import torch

from debiasing_methods.confidenceRegularization.overrides.loss import SmoothedDistillLoss
from debiasing_methods.confidenceRegularization.overrides.models import *
from debiasing_methods.confidenceRegularization.overrides.trainers import DistillerTrainer
from debiasing_methods.confidenceRegularization.utils import *

dirname = os.getcwd()


def choose_distill_model(model_name: str, loss_fn: ClfDistillLossFunction) -> AutoModelForQuestionAnswering:
    if model_name == 'bert-base-uncased':
        return DistillBertForQuestionAnswering.from_pretrained(model_name, loss_fn=loss_fn)
    elif model_name in ['roberta-base', 'roberta-large']:
        return DistillRobertaForQuestionAnswering.from_pretrained(model_name, loss_fn=loss_fn)
    elif model_name == 'electra-base-discriminator':
        return DistillElectraForQuestionAnswering.from_pretrained(model_name, loss_fn=loss_fn)

    raise NotImplementedError(f"Model: '{model_name}' is not supported. Please, modify 'choose_distill_model' method!")


def load_distill_preds(args, load_biased: bool) -> DataFrame:
    filename = get_preds_filename(args.model, args.bias, args.dataset, load_biased)
    print(filename)
    print(get_dataset_path(filename))
    return pd.read_json(get_dataset_path(filename))


def create_distill_dataset(train_dataset: Dataset, teacher_preds: DataFrame, bias_preds: DataFrame) -> Dataset:
    """
    Combine distillation examples (teacher_preds and bias_preds) into one dataset that can be fed through Trainer API
    :param train_dataset: Dataset in HF format
    :param teacher_preds: teacher predictions, DataFrame from JSON, loaded via load_distill_preds
    :param bias_preds: predictions of biased model, DataFrame from JSON, loaded via load_distill_preds
    :return Dataset
    """



    teacher_probs = teacher_preds['start_logits'] + teacher_preds['end_logits']
    if len(train_dataset['train']) + len(train_dataset['validation']) != len(teacher_probs):
        raise RuntimeError(f"Length of train_dataset is " +
                            f"{len(train_dataset['train']) + len(train_dataset['validation'])}, " +
                            f"but length of teacher_probs is {len(teacher_probs)} and should be equal")

    bias_probs = bias_preds['start_logits'] + bias_preds['end_logits']
    if len(train_dataset['train']) + len(train_dataset['validation']) != len(bias_probs):
        raise RuntimeError(f"Length of train_dataset is " +
                            f"{len(train_dataset['train']) + len(train_dataset['validation'])}, " +
                            f"but length of teacher_probs is {len(teacher_probs)} and should be equal")


    train_len = len(train_dataset['train'])
    valid_len = len(train_dataset['validation'])


    train_dataset['train'] = train_dataset['train'].add_column('teacher_probs_start',teacher_preds['start_logits'][:train_len])
    train_dataset['train'] = train_dataset['train'].add_column('teacher_probs_end', teacher_preds['end_logits'][:train_len])
    train_dataset['train'] = train_dataset['train'].add_column('bias_probs_start',bias_preds['start_logits'][:train_len])
    train_dataset['train'] = train_dataset['train'].add_column('bias_probs_end',bias_preds['end_logits'][:train_len])

    train_dataset['validation'] = train_dataset['validation'].add_column('teacher_probs_start', teacher_preds['start_logits'][train_len:train_len+valid_len])
    train_dataset['validation'] = train_dataset['validation'].add_column('teacher_probs_end', teacher_preds['end_logits'][train_len:train_len+valid_len])
    train_dataset['validation'] = train_dataset['validation'].add_column('bias_probs_start', bias_preds['start_logits'][train_len:train_len+valid_len])
    train_dataset['validation'] = train_dataset['validation'].add_column('bias_probs_end', bias_preds['end_logits'][train_len:train_len+valid_len])

    return train_dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", default="bert-base-uncased", type=str,
                        help="Model to be debiased")
    parser.add_argument("--dataset", default="squad", type=str,
                        help="HuggingFace dataset name, e.g.: 'squad'")
    parser.add_argument("--bias", default="distances", type=str,
                        help="On which bias to train model. Supports all biases of 'isbiased' lib. "
                             "Possible values: 'similar_words','distances','kth_sentence','cosine_similarity',"
                             "'answer_length','max_sim_ents','answer_subject_positions'")
    parser.add_argument("--output_dir",
                        default="./results",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    # parser.add_argument("--preds_dir",
    #                     default="./dataset",
    #                     type=str,
    #                     help="Directory to save teacher predictions to.")
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
    debiased_name = "debiased-conf_reg-" + model_checkpoint
    model_save_path = os.path.join(dirname, 'saved_models', debiased_name)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    loss_fn = SmoothedDistillLoss()
    distilled_model = choose_distill_model(args.model, loss_fn)
    distilled_model.to(device)

    # load teacher predictions and biased predictions
    dataset = load_dataset(args.dataset)
    ### for testing
    dataset = dataset['validation'].train_test_split(train_size=100)
    dataset = dataset['train'].train_test_split(train_size=0.8)
    dataset['validation'] = dataset['test']
    del dataset['test']
    ###
    tokenized_squad = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names,
                                  fn_kwargs={'tokenizer': tokenizer, 'args': args})
    ### for testing
    tokenized_squad['train'] = tokenized_squad['train'].select(range(80))
    tokenized_squad['validation'] = tokenized_squad['validation'].select(range(20))
    ###

    print("Got dataset...")
    data_collator = DefaultDataCollator()

    # create distilled dataset here
    # merge tokenized dataest with parent_preds, biased_preds
    teacher_preds = load_distill_preds(args, False)
    biased_preds = load_distill_preds(args, True)
    distil_dataset = create_distill_dataset(tokenized_squad, teacher_preds, biased_preds)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
    )

    # TODO - not working now!
    # need to override Trainer, because of distillation - model's loss function needs input of teacher_preds and biased_preds
    # MAYBE POSSIBLE TO COMBINE PREDICTIONS INTO ONE TRAIN DATASET - create_distill_dataset
    trainer = DistillerTrainer(
        model=distilled_model,
        args=training_args,
        train_dataset=distil_dataset["train"],
        eval_dataset=distil_dataset["validation"],
        # teacher_preds = load json from
        #   debiasing_methods/confidenceRegularization/dataset/teacher_preds_bert-base-uncased_finetuned_baseline.json
        #   pass somehow, so that model is fed with train_dataset
        teacher_predictions=None,
        # biased_preds = same as teacher_preds
        #   debiasing_methods/confidenceRegularization/dataset/teacher_preds_*.json
        biased_predictions=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(debiased_name)


if __name__ == '__main__':
    main()
