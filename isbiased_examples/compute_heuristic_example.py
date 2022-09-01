from isbiased.bias_significance import BiasSignificanceMeasure
import pandas as pd
from datasets import load_dataset

# provided dataset needs to be in squad format and contain columns 'id', 'title', 'context', 'question', 'answers'
datasets = load_dataset("squad")

# when creating an object, you can optionally set values for iterations (default 100) and sample_size (default 800) 
bias_significance = BiasSignificanceMeasure()

# you can use local folder with finetuned model or some qa model from huggingface
model_path = 'distilbert-base-uncased-distilled-squad' #path to local folder with fine-tuned model
# bert base model fine-tuned on squad dataset from huggingface
# model_path = 'csarron/bert-base-uncased-squad-v1'

# at first, we need to get predictions for our provided model and dataset, the function also computes metrics - exact match and f1
# predictions will be added to the internal class DataFrame 
# function also returns dataset with predictions
metrics, dataset = bias_significance.evaluate_model_on_dataset(model_path, datasets['validation']) 

# for computation of selected heuristic, use function compute_heuristic()
# the function takes parameter dataset, which is pandas DataFrame, 
# and heuristic, which is the name of selected heuristic, there will be also column with similar name added to the data
# the value for heuristic can be one of the following:
# 'similar_words'
# 'distances'
# 'kth_sentence'
# 'cosine_similarity'
# 'answer_length'
# 'max_sim_ents'
# 'answer_subject_positions'
heuristic = 'distances'
bias_significance._compute_heuristic(pd.DataFrame(dataset), heuristic)

