
from heuristics import *

import json
import transformers
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import default_data_collator
from datasets import load_dataset, load_metric
import collections
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import ClassLabel, Sequence
from tqdm.auto import tqdm
import os
from statistics import mean

class BiasSignificanceMeasure:
    
    num_for_average_metrics = 1
    metric = load_metric("squad")


    def __init__(self, data: Dataset, iterations = 100, sample_size = 800):
        self.data = pd.DataFrame(data) 
        self.iterations = iterations
        self.sample_size = sample_size
    

    def _compute_metrics_for_sample(self, sample):
        """Computation of metrics for dataset sample
        Computes exact match and F1 between predicted and ground truth answers

        Args:
            sample (Dataframe): sample from dataset

        Returns:
            dict: computed metrics
        """
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in zip(sample['id'], sample['prediction_text'])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in sample]
        return self.metric.compute(predictions=formatted_predictions, references=references)


    def _compute_metrics_for_bunch(self, data):
        """Sampling the dataset for specified number of iterations and computation of metrics for samples

        Args:
            data (Pandas Dataframe): dataset with computed heuristics

        Returns:
            Pandas Dataframe: computed heuristics for samples
        """
        exact_list = []
        f1_list = []

        for i in tqdm(range(self.iterations)):
            df = data.sample(n=self.sample_size)
            sample = Dataset.from_pandas(df)
            metrics1 = self._compute_metrics_for_sample(sample)
            exact_list.append(metrics1['exact_match'])
            f1_list.append(metrics1['f1'])
        
        d = {'exact_match': exact_list, 'f1': f1_list}
        df = pd.DataFrame(d)

        return df


    def _find_the_distance_between_intervals(self, lower_025, lower_975, higher_025, higher_975):
        """Detect if there is some interval overlap between the two intervals or if there is not
        If there is not an overlap, it computes the distance between 2.5% and 97.5% quantiles

        Args:
            lower_025 (int): 2.5% quantile for lower subset
            lower_975 (int): 97.5% quantile for higher subset
            higher_025 (int): 2.5% quantile for lower subset
            higher_975 (int): 97.5% quantile for higher subset

        Returns:
            bool, decimal: boolean flag and the distance between intervals
        """
        distance_between_intervals = 0
        if lower_975 > higher_025 and lower_025 > higher_975:
            distance_between_intervals = lower_025 - higher_975
            return True, distance_between_intervals 
        elif higher_975 > lower_025 and higher_025 > lower_975:
            distance_between_intervals = higher_025 - lower_975
            return True, distance_between_intervals
        else:
            return False, distance_between_intervals
    

    def _compute_metrics_average_split(self, heuristic, threshold):
        """Function which calls the previous ones and provide dataset splits and logging into files

        Args:
            heuristic (str): column of the dataset based on which we want to split the data
            threshold (decimal): number for the split of dataset, the values lower or equal will be in one subset and values higher than the threshold will be in the other

        Returns:
            decimal, decimal: distances for both metrics, exact match and F1
        """

        if heuristic == 'distances':
            data_higher, data_lower = [x for _, x in self.data[self.data.distances >= 0].groupby(self.data[heuristic] <= threshold)]
        elif heuristic == 'answer_subject_positions':
            data_higher, data_lower = [x for _, x in self.data[self.data.answer_subject_positions >= 0].groupby(self.data[heuristic] <= threshold)]
        else:            
            data_higher, data_lower = [x for _, x in self.data.groupby(self.data[heuristic] <= threshold)]

        if len(data_higher) < self.sample_size or len(data_lower) < self.sample_size:
            return [-1, -1, 0, 0, 0, 0]

        lower_exact_match_quantile_025 = []
        lower_exact_match_quantile_975 = []
        lower_exact_match_mean = []
        lower_f1_quantile_025 = []
        lower_f1_quantile_975 = []
        lower_f1_mean = []
        higher_exact_match_quantile_025 = []
        higher_exact_match_quantile_975 = []
        higher_exact_match_mean = []
        higher_f1_quantile_025 = []
        higher_f1_quantile_975 = []
        higher_f1_mean = []
        df_lower = []
        df_higher = []


        for i in tqdm(range(self.num_for_average_metrics)):
            df_lower = self._compute_metrics_for_bunch(data_lower)
            lower_exact_match_quantile_025.append(df_lower['exact_match'].quantile(0.025))
            lower_exact_match_quantile_975.append(df_lower['exact_match'].quantile(0.975))
            lower_exact_match_mean.append(df_lower['exact_match'].mean())
            lower_f1_quantile_025.append(df_lower['f1'].quantile(0.025))
            lower_f1_quantile_975.append(df_lower['f1'].quantile(0.975))
            lower_f1_mean.append(df_lower['f1'].mean())

            
            df_higher = self._compute_metrics_for_bunch(data_higher)
            higher_exact_match_quantile_025.append(df_higher['exact_match'].quantile(0.025))
            higher_exact_match_quantile_975.append(df_higher['exact_match'].quantile(0.975))
            higher_exact_match_mean.append(df_higher['exact_match'].mean())
            higher_f1_quantile_025.append(df_higher['f1'].quantile(0.025))
            higher_f1_quantile_975.append(df_higher['f1'].quantile(0.975))
            higher_f1_mean.append(df_higher['f1'].mean())

           
        is_not_overlap_em, distance_em = self._find_the_distance_between_intervals(mean(lower_exact_match_quantile_025), mean(lower_exact_match_quantile_975), mean(higher_exact_match_quantile_025), mean(higher_exact_match_quantile_975))
        is_not_overlap_f1, distance_f1 = self._find_the_distance_between_intervals(mean(lower_f1_quantile_025), mean(lower_f1_quantile_975), mean(higher_f1_quantile_025), mean(higher_f1_quantile_975))


        # print(f"Average exact match with params: samples {self.sample_size} iters {self.iterations} ---- are independent: {is_not_overlap_em} the distance is: {distance_em}")

        # print(f"Average f1 with params: samples {self.sample_size} iters {self.iterations} ---- are independent: {is_not_overlap_f1} the distance is: {distance_f1}")

        return [distance_em, distance_f1, len(data_lower), len(data_higher), mean(lower_exact_match_quantile_975), mean(higher_exact_match_quantile_975)]
    

    def _find_best_threshold_for_heuristic(self, distances_dictionary: dict):
        best_threshold = -1
        max_distance = -1
        size_of_smaller = 0

        for key, value in zip(distances_dictionary.keys(), distances_dictionary.values()):
            if (value[0] > max_distance and max_distance == -1)  or (max_distance != -1 and (value[0] > max_distance or (value[0] > 0 and size_of_smaller < self.sample_size*2)) and value[2] > self.sample_size*2 and value[3] > self.sample_size*2):
                max_distance = value[0]
                best_threshold = key
                size_of_smaller = value[2] if value[2] < value[3] else value[3]
        
        return best_threshold, max_distance, distances_dictionary


    def find_longest_distance(self, heuristic: str):
        """Finds out the longest distance between intervals for bunch of thresholds

        Args:
            heuristic (str): Dataframe column
            low_bound (int): the number from which the threshold will start from
            upp_bound (int): the number for which the threshold will go -1
        """
        index_em = 0
        index_f1 = 0
        max_em_distance = 0
        max_f1_distance = 0
        distances_dict = {}

        distance_em = 0
        distance_f1 = 0

        min_value_for_threshold = int(self.data[heuristic].min()) + 1 if self.data[heuristic].min() > 0 else 0
        max_value_for_threshold = self.data[heuristic].max()

        # print(f"Min value: {min_value_for_threshold} and max value: {max_value_for_threshold}")

        while(min_value_for_threshold < max_value_for_threshold):
            # distance_em, distance_f1 = self._compute_metrics_average_split(heuristic, min_value_for_threshold)
            distances_dict[min_value_for_threshold] = self._compute_metrics_average_split(heuristic, min_value_for_threshold)
            if distance_em > max_em_distance:
                # max_em_distance = distance_em
                max_em_distance = distances_dict.get(min_value_for_threshold)[0]
                index_em = min_value_for_threshold
            if distance_f1 > max_f1_distance:
                # max_f1_distance = distance_f1
                max_f1_distance = distances_dict.get(min_value_for_threshold)[1]
                index_f1 = min_value_for_threshold

            # distances_dict[min_value_for_threshold] = [distance_em, distance_f1]

            if max_value_for_threshold > 1:
                min_value_for_threshold += 1
            else:
                min_value_for_threshold += 0.1

        # print(f"The biggest distance between exact match intervals was with threshold {index_em} and the distance was {max_em_distance}.")
        # print(f"The biggest distance between f1 intervals was with threshold {index_f1} and the distance was {max_f1_distance}.")

        return self._find_best_threshold_for_heuristic(distances_dict)


    def evaluate_model_on_dataset(self, model_path:str, dataset_eval: Dataset):
        squad_v2 = False
        model_checkpoint = "roberta-large"
        m_name = model_path.split("/")[-1]
        # saved_model = 'models/roberta-large-finetuned-squad_with_callbacks_baseline'
        # saved_model = model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        data_collator = default_data_collator
        metric = load_metric("squad_v2" if squad_v2 else "squad")

        pad_on_right = tokenizer.padding_side == "right"
        max_length = 384
        doc_stride = 128

        model_name = model_checkpoint.split("/")[-1]

        trainer = Trainer(
            model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

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
        def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
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

                min_null_score = None # Only used if squad_v2 is True.
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
                    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                    
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

        
        def model_evaluation_on_dataset(dataset_eval, save_dataframe_with_predictions = False, name = 'model_name'):
            """Model evaluation on specific dataset
            Calls previous functions and evaluate the dataset on the model for exact match and F1

            Args:
                dataset (Dataset): validation dataset
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
            
            validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))

            final_predictions = postprocess_qa_predictions(dataset_eval, validation_features, raw_predictions.predictions)

            predictions = [v for k, v in final_predictions.items()]
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset_eval]
            
            if save_dataframe_with_predictions:
                data = pd.DataFrame(dataset_eval)   
                data['prediction_text'] = predictions
                data.to_json('./datasets/' + name + '.json', orient='records')

            return metric.compute(predictions=formatted_predictions, references=references), data
        
        metrics, self.data = model_evaluation_on_dataset(dataset_eval, save_dataframe_with_predictions = True, name = m_name)
        
        return metrics


    def compute_heuristics(self):
        # train_dataset = pd.read_json('./datasets/squad_train.json')
        train_dataset = pd.DataFrame(load_dataset("squad")['train'])

        squad_with_heuristics = ComputeHeuristics(self.data, train_dataset)
        squad_with_heuristics.compute_all_heuristics()
        self.data = squad_with_heuristics.data

    
    def compute_heuristic(self, heuristic: str):
        train_dataset = pd.DataFrame(load_dataset("squad")['train'])

        computed_heuristic = ComputeHeuristics(self.data, train_dataset)
        computed_heuristic.compute_heuristic(heuristic)
        self.data = computed_heuristic.data


    
    def split_data_by_heuristics(self, dataset: Dataset, heuristic: str):
        best_threshold, distance, dist_dict = self.find_longest_distance(heuristic)
        # print(best_threshold)
        # print(distance)
        # print(dist_dict)

        if best_threshold != -1:
            comp_heuristic = ComputeHeuristics(pd.DataFrame(dataset), pd.DataFrame(dataset))
            comp_heuristic.compute_heuristic(heuristic)
            dataset = comp_heuristic.data

            if dist_dict.get(best_threshold)[4] > dist_dict.get(best_threshold)[5]:
                unbiasedDataset, biasedDataset = [x for _, x in dataset.groupby(dataset[heuristic] <= best_threshold)]
            else:
                biasedDataset, unbiasedDataset = [x for _, x in dataset.groupby(dataset[heuristic] <= best_threshold)]
        
        else:
            print('Dataset was not split!!!')
            return (dataset, dataset)

        return (Dataset.from_pandas(biasedDataset), Dataset.from_pandas(unbiasedDataset))