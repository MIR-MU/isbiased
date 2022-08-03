from bias_significance import *
import pandas as pd
from datasets import Dataset

datasets = load_dataset("squad")

# when creating an object, dataset needs to be provided, you can optionally set values for iterations (default 100) and sample_size (default 800) 
# in this example, squad validation is used
# provided dataset needs to be in squad format and contain columns 'id', 'title', 'context', 'question', 'answers'

bias_significance = BiasSignificanceMeasure(datasets['validation'].select(range(2000)))

# you can use local folder with finetuned model or some qa model from huggingface
model_path = '/models/electra-base-discriminator-finetuned-squad_with_callbacks_baseline'  #path to local folder with fine-tuned model
# bert base model fine-tuned on squad dataset from huggingface
# model_path = 'csarron/bert-base-uncased-squad-v1'

# at first, we need to get predictions for our provided model and dataset, the function also computes metrics - exact match and f1
# predictions will be added to the internal class DataFrame 
bias_significance.evaluate_model_on_dataset(model_path, datasets['validation'].select(range(2000)))

# function find_longest_distance() does the measuring of bias significance based on selected heuristic for every possible threshold
# it can be used to find the best threshold - the one with highest distance between intervals
# the function takes one parameter, heuristic, which is the name of selected heuristic
# the function returns the best threshold, the maximum exact match distance and dictionary containing data for every threshold
# the value for heuristic can be one of the following:
# 'similar_words'
# 'distances'
# 'kth_sentence'
# 'cosine_similarity'
# 'answer_length'
# 'max_sim_ents'
# 'answer_subject_positions'
heuristic = 'distances'

best_threshold, max_distance, distances_dictionary = bias_significance.find_longest_distance(heuristic)

print((best_threshold, max_distance, distances_dictionary))

biasedDataset, unbiasedDataset = bias_significance.split_data_by_heuristics(datasets['train'], heuristic)

print((biasedDataset, unbiasedDataset))
