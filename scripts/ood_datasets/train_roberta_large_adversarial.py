import collections
import shutil
from typing import List

import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import default_data_collator


print(transformers.__version__)
# model_checkpoint = "bert-base-uncased"
# model_checkpoint = "roberta-base"
model_checkpoint = "roberta-large"
dataset_name = 'newsqa'
# model_checkpoint = "google/electra-base-discriminator"
batch_size = 16
squad_v2 = False
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.

if dataset_name == "squad":
    datasets = load_dataset("squad_v2" if squad_v2 else "squad")


if dataset_name == 'triviaqa':
    datasets = DatasetDict({"train":Dataset.from_pandas(pd.read_json('./datasets/triviaqa_train_formatted.json')),
                            "validation":Dataset.from_pandas(pd.read_json('./datasets/triviaqa_dev_formated.json'))})


if dataset_name == 'nq':
    datasets = DatasetDict({"train":Dataset.from_pandas(pd.read_json('./datasets/nq_train_formatted.json')),
                            "validation":Dataset.from_pandas(pd.read_json('./datasets/nq_dev_formated.json'))})


if dataset_name == 'adversarial':
    datasets = load_dataset('adversarial_qa', 'adversarialQA')


def get_answers_from_spans(spans: str, context: str) -> List[str]:
    answers_coords = spans.split(",")
    out = []
    for span in answers_coords:
        span_start, span_end = span.split(":")
        span_start, span_end = int(span_start), int(span_end)
        span_text = " ".join(context.split()[span_start: span_end])
        out.append(span_text)

    return out


if dataset_name == 'newsqa':
    datasets = load_dataset("boyiwei/newsqa")
    for k in datasets:
        # d[k]["id"] = d[k]["story_id"]
        # d[k]["context"] = d[k]["story_text"]
        datasets[k] = datasets[k].map(lambda x: {"id": x["story_id"],
                                                 "context": x["story_text"],
                                                 "answers": {"text": get_answers_from_spans(x["answer_token_ranges"],
                                                                                            x["story_text"])}
                                                 })
        datasets[k] = datasets[k].map(lambda x: {"answers": {"text": [a for a in x["answers"]["text"]],
                                                             "answer_start": [x["story_text"].find(a) for a in
                                                                              x["answers"]["text"]]
                                                             }
                                                 })

if dataset_name == 'searchqa':
    # these datasets were produced using convert_from_text_format.py script
    datasets = DatasetDict({
        "train": Dataset.from_pandas(pd.read_feather("searchqa_tmp_train.feather")),
        "validation": Dataset.from_pandas(pd.read_feather("searchqa_tmp_validation.feather"))
    })
    datasets["train"] = datasets["train"].map(lambda x: {"answers": {"text": [a for a in x["answers"]],
                                                                     "answer_start": [x["context"].find(a) for a in x["answers"]]}
                                                         })
    datasets["validation"] = datasets["validation"].map(lambda x: {"answers": {"text": [a for a in x["answers"]],
                                                                               "answer_start": [x["context"].find(a) for a in x["answers"]]}
                                                                   })


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pad_on_right = tokenizer.padding_side == "right"


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


import os

os.environ["WANDB_DISABLED"] = "true"


# Preprocessing function for validation dataset from the HuggingFace Jupyter notebook
# with original comments
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# Postprocessing function from the HuggingFace Jupyter notebook
# with original comments
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if len(offset_mapping[start_index]) == 0 or len(offset_mapping[end_index]) == 0:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


metric = load_metric("squad_v2" if squad_v2 else "squad")


def training():
    tokenized_datasets = datasets.map(prepare_train_features, batched=True,
                                      remove_columns=datasets["train"].column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
            f"{model_name}-finetuned-{dataset_name}_with_callbacks",
            #     evaluation_strategy = "epoch",
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            logging_steps=200,
            save_total_limit=5,
            learning_rate=2e-5,
            #     warmup_ratio=0.1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            #         max_steps=20,
            weight_decay=0.01,
            report_to="none",
            load_best_model_at_end=True,
            #     push_to_hub=False,
            no_cuda=True,
    )

    data_collator = default_data_collator

    trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()

    trainer.save_model(f"{model_name}-finetuned-squad_with_callbacks_{type_}")

    if not os.path.exists('./saved_finetuned_models/'):
        os.mkdir('./saved_finetuned_models/')

    shutil.make_archive(f"./saved_finetuned_models/{model_name}-finetuned-{dataset_name}_with_callbacks_{type_}", 'zip',
                        f"{model_name}-finetuned-{dataset_name}_with_callbacks_{type_}")

    return trainer


def model_evaluation_on_dataset(dataset_eval, trainer, save_dataframe_with_predictions=False, name='model_name'):
    """Model evaluation on specific dataset
    Calls previous functions and evaluate the dataset on the model for exact match and F1

    Args:
        dataset_eval (Dataset): validation dataset
        save_dataframe_with_predictions (bool, optional): flag for saving the dataset with prediction. Defaults to False.
        name (str, optional): name of the model. Defaults to 'model_name'.

    Returns:
        _type_: _description_
    """
    validation_features = dataset_eval.map(
            prepare_validation_features,
            batched=True,
            remove_columns=dataset_eval.column_names
    )

    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(type=validation_features.format["type"],
                                   columns=list(validation_features.features.keys()))

    final_predictions = postprocess_qa_predictions(dataset_eval, validation_features, raw_predictions.predictions)

    predictions = [v for k, v in final_predictions.items()]
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset_eval]

    if save_dataframe_with_predictions:
        if not os.path.exists('./data_with_predictions/'):
            os.mkdir('./data_with_predictions/')

        data_with_predictions = pd.DataFrame(dataset_eval)
        data_with_predictions['prediction_text'] = predictions
        data_with_predictions.to_json('./' + name + '-with-predictions.json',
                                      orient='records')  # data_with_predictions/

    return metric.compute(predictions=formatted_predictions, references=references)


trainer = training()

print("Done")
