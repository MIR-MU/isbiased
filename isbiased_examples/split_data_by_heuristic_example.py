from isbiased.bias_significance import *

# provided dataset needs to be in squad format and contain columns 'id', 'title', 'context', 'question', 'answers'
datasets = load_dataset("squad")

# when creating an object, you can optionally set values for iterations (default 100) and sample_size (default 800) 
bias_significance = BiasSignificanceMeasure()

# you can use local folder with finetuned model or some qa model from huggingface
# model_path = 'models/roberta-base-finetuned-squad_with_callbacks_baseline' #path to local folder with fine-tuned model
# model_path = 'bert-base-multilingual-cased' #path to local folder with fine-tuned model
# bert base model fine-tuned on squad dataset from huggingface
model_path = 'csarron/bert-base-uncased-squad-v1'

# at first, we need to get predictions for our provided model and dataset, the function also computes metrics - exact match and f1
# predictions will be added to the internal class DataFrame 
# function also returns dataset with predictions
metrics, dataset = bias_significance.evaluate_model_on_dataset(model_path, datasets['validation'].select(range(2000)))

# for computation of selected heuristic, use function compute_heuristic()
# the function takes parameter heuristic, which is the name of selected heuristic, there will be also column with similar name added to the data
# the value for heuristic can be one of the following:
# 'similar_words'
# 'distances'
# 'kth_sentence'
# 'cosine_similarity'
# 'answer_length'
# 'max_sim_ents'
# 'answer_subject_positions'
heuristic = 'distances'

# when you have computed heuristic and predictions
# you can split dataset of your choice based on selected heuristic
# in the function the best threshold for selected heuristic is computed, then the heuristic is computed for provided dataset and data are split
# the function returns two Dataset objects, first with biased and second with unbiased data
# in this example, squad train is used and selected heuristic is 'distances'
biasedDataset, unbiasedDataset = bias_significance.split_data_by_heuristics(dataset, datasets['train'], heuristic)

print(biasedDataset)
