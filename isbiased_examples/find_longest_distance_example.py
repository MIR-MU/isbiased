from isbiased.bias_significance import *

# provided dataset needs to be in squad format and contain columns 'id', 'title', 'context', 'question', 'answers'
datasets = load_dataset("squad")

# when creating an object, you can optionally set values for iterations (default 100) and sample_size (default 800) 
bias_significance = BiasSignificanceMeasure()

# you can use local folder with finetuned model or some qa model from huggingface
# the script will infer autmatically infer the correct inference implementation among extractive and generative QA

# bert base model fine-tuned on squad dataset from huggingface
model_path = 'csarron/bert-base-uncased-squad-v1'
# t5 model fine-tuned for generative question answering
# model_path = 'sjrhuschlee/flan-t5-base-squad2'

# at first, we need to get predictions for our provided model and dataset, the function also computes metrics - exact match and f1
# predictions will be added to the internal class DataFrame 
# function also returns dataset with predictions
metrics, dataset = bias_significance.evaluate_model_on_dataset(model_path, datasets['validation'].select(range(2000)))

# function find_longest_distance() does the measuring of bias significance based on selected heuristic for every possible threshold
# it can be used to find the best threshold - the one with highest distance between intervals
# the function takes two parameters, dataset, which is Dataset object, 
# and heuristic, which is the name of selected heuristic
# the function returns the best threshold, the maximum exact match distance and dictionary containing data for every threshold, 
# and dataset
# the value for heuristic can be one of the following:
# 'similar_words'
# 'distances'
# 'kth_sentence'
# 'cosine_similarity'
# 'answer_length'
# 'max_sim_ents'
# 'answer_subject_positions'
heuristic = 'distances'

threshold_distance_dictionary, dataset = bias_significance.find_longest_distance(dataset, heuristic)
best_threshold, max_distance, distances_dictionary = threshold_distance_dictionary

print((best_threshold, max_distance, distances_dictionary))

if max_distance != -1:
    print("Found statistically significant reliance on bias "
          "making a difference in performance of at least %s points %s percent of performance" % (max_distance, ))
    biasedDataset, unbiasedDataset = bias_significance.split_data_by_heuristics(dataset, datasets['train'], heuristic)

    print((biasedDataset, unbiasedDataset))
else:
    print("Could not find statistically significant reliance on bias.")

