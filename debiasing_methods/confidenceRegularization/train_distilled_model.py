import argparse

import pandas as pd
from datasets import load_dataset, Dataset
from pandas import DataFrame
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, \
    PreTrainedModel, EarlyStoppingCallback

from debiasing_methods.confidenceRegularization.overrides.loss import SmoothedDistillLoss
from debiasing_methods.confidenceRegularization.overrides.models import *
from debiasing_methods.confidenceRegularization.overrides.trainers import DistillerTrainer
from debiasing_methods.confidenceRegularization.utils import *

dirname = os.getcwd()


def choose_distill_model(model_name: str, loss_fn: ClfDistillLossFunction) -> AutoModelForQuestionAnswering:
    if "bert-base" in model_name:
        return DistilBertForQuestionAnswering.from_pretrained(model_name, loss_fn=loss_fn)
    elif model_name in ['roberta-base', 'roberta-large']:
        return DistillRobertaForQuestionAnswering.from_pretrained(model_name, loss_fn=loss_fn)
    elif model_name == 'electra-base-discriminator':
        return DistillElectraForQuestionAnswering.from_pretrained(model_name, loss_fn=loss_fn)

    raise NotImplementedError(f"Model: '{model_name}' is not supported. Please, modify 'choose_distill_model' method!")


def load_distill_preds(args, load_biased: bool) -> DataFrame:
    filename = get_preds_filename(args.trained_model, args.bias, args.dataset, load_biased)
    print(filename)
    print(get_dataset_path(filename))
    return pd.read_json(get_dataset_path(filename))


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


def create_distill_dataset(dataset: Dataset, teacher: PreTrainedModel, bias: PreTrainedModel) -> Dataset:
    """
    Combine distillation examples (teacher_preds and bias_preds) into one dataset that can be fed through Trainer API
    :param dataset: Dataset in HF format
    :param teacher: Teacher QA model to distil from
    :param bias: Biased QA model to downscale the Teacher's prediction with
    :return Dataset
    """

    teacher_start_logits, teacher_end_logits = infer_model_start_end_logits(teacher, dataset["train"])
    bias_start_logits, bias_end_logits = infer_model_start_end_logits(bias, dataset["train"])

    dataset['train'] = dataset['train'].add_column('teacher_probs_start', teacher_start_logits.tolist())
    dataset['train'] = dataset['train'].add_column('teacher_probs_end', teacher_end_logits.tolist())
    dataset['train'] = dataset['train'].add_column('bias_probs_start', bias_start_logits.tolist())
    dataset['train'] = dataset['train'].add_column('bias_probs_end', bias_end_logits.tolist())

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--trained_model", default="bert-base-cased", type=str,
                        help="Model to be debiased")
    parser.add_argument("--biased_model", default="bert-base-cased", type=str,
                        help="Pre-trained biased model")
    parser.add_argument("--teacher_model", default="bert-base-cased", type=str,
                        help="Pre-trained full model")
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
                        default=32,
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

    model_checkpoint = args.trained_model
    print("Model:   ", model_checkpoint)
    print("Bias model:   ", args.biased_model)
    print("Teacher model:   ", args.teacher_model)
    debiased_name = "debiased-conf_reg-" + model_checkpoint

    model_save_path = os.path.join(dirname, 'saved_models', debiased_name)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    loss_fn = SmoothedDistillLoss()
    distilled_model = choose_distill_model(args.trained_model, loss_fn)
    distilled_model.to(device)

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

    bias_model = AutoModelForQuestionAnswering.from_pretrained(args.biased_model)
    teacher_model = AutoModelForQuestionAnswering.from_pretrained(args.teacher_model)

    distil_dataset = create_distill_dataset(tokenized_squad, teacher_model, bias_model)

    training_args = TrainingArguments(
            output_dir=model_save_path,
            evaluation_strategy="steps",
            eval_steps=1000,  # Evaluation and Save happens every 200 steps
            save_steps=1000,
            logging_steps=1000,
            save_total_limit=50,  # Only last 10 models are saved. Older ones are deleted.
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            load_best_model_at_end=True,
    )

    trainer = DistillerTrainer(
        model=distilled_model,
        args=training_args,
        train_dataset=distil_dataset["train"],
        eval_dataset=distil_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
    )

    trainer.train()

    trainer.save_model(debiased_name)


if __name__ == '__main__':
    main()
