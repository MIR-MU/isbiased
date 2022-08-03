# IsBiased
A library to measure biases of question answering models.

### How to use

First, install:

```shell
pip install git+https://github.com/MIR-MU/isbiased.git
```
or for further development:
```shell
git clone -b {this branch} git+https://github.com/MIR-MU/isbiased.git
pip install -e isbiased
```

Instantiate the main `BiasSignificanceMeasure` class on a specified dataset(?) in HF format(?):
```python
from isbiased.bias_significance import BiasSignificanceMeasure
from datasets import load_dataset

# provided dataset needs to be in squad format and contain columns 'id', 'title', 'context', 'question', 'answers'
squad = load_dataset("squad")

# full dataset - takes longer:
measurer = BiasSignificanceMeasure(squad['train'])
# or subset - faster:
measurer = BiasSignificanceMeasure(squad['train'].select(range(2000)))
```

Then, you can do one, or more of the following:

1. Measure a bias of your selected QA model ([full example](isbiased_examples/find_longest_distance_example.py)):
```python
# you can use local folder with finetuned model or some qa model from huggingface
model_path = "path/to/pretrained/HF/QA/model"

# We implement heuristics for the following bias_id:
# 'similar_words'
# 'distances'
# 'kth_sentence'
# 'cosine_similarity'
# 'answer_length'
# 'max_sim_ents'
# 'answer_subject_positions'
bias_id = "distances"

# at first, we need to get predictions for our provided model and dataset, the function also computes metrics - exact match and f1
# predictions will be added to the internal class DataFrame 
bias_significance.evaluate_model_on_dataset(model_path, squad['validation'])

bias_significance.compute_heuristic(bias_id)  # compute heuristic
best_heuristic_threshold, bias_significance, distances_dict = bias_significance.find_longest_distance(heuristic)

print("Model '%s' bias '%s' significance is %s" % (model_path, bias_id, bias_significance))
```

2. Split data according to the selected heuristic to most 
significantly differentiate the model's performance ([full example](isbiased_examples/split_data_by_heuristic_example.py)):
```python
# after running bias_significance.compute_heuristic(bias_id)
# that finds the best threshold for selected heuristic is computed, then the heuristic is computed for provided dataset and data are split,
# run:
biasedDataset, unbiasedDataset = bias_significance.split_data_by_heuristics(datasets['train'], heuristic)
# biasedDataset contain biased (=better-performing samples) and unbiasedDataset contain unbiased (=worse-performing) data
# biasedDataset can be used e.g. to train a "bias model"
```

