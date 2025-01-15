from bias_significance import *
import pandas as pd
from datasets import Dataset

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
metrics, dataset = bias_significance.evaluate_model_on_dataset(model_path, datasets['validation']) 