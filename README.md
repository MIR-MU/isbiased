# IsBiased
IsBiased is a library implementing a metric and heuristics for assessing language models' reliance on known prediction shortcuts in question-answering. 
It is based on a maximum difference between two segments of data obtained from splitting the input dataset on one of the
known bias features.
Further details can be found in our EACL 2024 paper [Think Twice: Measuring the Efficiency of Eliminating Prediction Shortcuts of Question Answering Models](https://aclanthology.org/2024.eacl-long.133/). 

Feel free to reach out if you have any questions or suggestions!

### How to use

First, install:

```shell
pip install git+this_repo.git
```
or for further development:
```shell
git clone -b {this branch} git+this_repo/isbiased.git
pip install -e isbiased
```

Instantiate the main `BiasSignificanceMeasure` class:
```python
from isbiased.bias_significance import BiasSignificanceMeasure
from datasets import load_dataset

# provided dataset needs to be in squad format and contain columns 'id', 'title', 'context', 'question', 'answers'
squad = load_dataset("squad")

measurer = BiasSignificanceMeasure()

```

Then, you can do one, or more of the following:

1. Measure a bias of your selected QA model ([full example](isbiased_examples/find_longest_distance_example.py)):
```python
# you can use local folder with finetuned model or some qa model from huggingface
# model_path = "path/to/pretrained/HF/QA/model"
# e.g. the most popular QA model on HF:
model_path = "deepset/roberta-base-squad2"

# We implement heuristics for the following bias_id:
# 'similar_words'
# 'distances'
# 'kth_sentence'
# 'cosine_similarity'
# 'answer_length'
# 'max_sim_ents'
# 'answer_subject_positions'
bias_id = "distances"

# first, we need to get predictions for our provided model and dataset, the function also computes metrics - exact match and f1
# predictions will be added to the internal class DataFrame 
metrics, dataset = measurer.evaluate_model_on_dataset(model_path, squad['validation'])

threshold_distance_dictionary, dataset = measurer.find_longest_distance(dataset, bias_id)
best_heuristic_threshold, bias_significance, distances_dict = threshold_distance_dictionary

print("Model '%s' bias '%s' significance is %s" % (model_path, bias_id, bias_significance))
```

2. Split data according to the selected heuristic to most 
significantly differentiate the model's performance ([full example](isbiased_examples/split_data_by_heuristic_example.py)):
```python
# after running the commands above, measurer has found the optimal configuration of selected heuristic,
# and the configured heuristic can be applied to split the data within an arbitrary dataset
# run:
biased_samples, non_biased_samples = measurer.split_data_by_heuristics(dataset, squad['train'], bias_id)
# biased_samples contains biased (=better-performing) and non_biased_samples contain unbiased (=worse-performing) samples of input dataset
# resulting data segments can be used, e.g. to train a "bias model", or to balance the train set and mitigate reliance on bias
```

### Citation

If you use our library in your research, please cite the corresponding EACL 2024 paper [Think Twice: Measuring the Efficiency of Eliminating Prediction Shortcuts of Question Answering Models](https://aclanthology.org/2024.eacl-long.133/):

```bibtex
@inproceedings{mikula-etal-2024-think,
    title = "Think Twice: Measuring the Efficiency of Eliminating Prediction Shortcuts of Question Answering Models",
    author = "Mikula, Luk{\'a}{\v{s}}  and
      {\v{S}}tef{\'a}nik, Michal  and
      Petrovi{\v{c}}, Marek  and
      Sojka, Petr",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.133/",
    pages = "2179--2193",
}
```

