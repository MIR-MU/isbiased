# IsBiased

A library to quantify _Bias significance_ of question answering models.
Intended use of this library is to complement end-to-end evaluations of the models (such as out-of-distribution performance)
for a finer-grained report on the mitigation of known model biases, commonly addressed by the methods for more robust training.

## How to use

First, install the library (tested with python3.8):

```shell
git clone git+https://github.com/repository/isbiased.git
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
model_path = "path/to/your/pretrained/HF/QA/model"
# for instance, you can use:
model_path = "ahotrod/electra_large_discriminator_squad2_512"

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
metrics, dataset = measurer.evaluate_model_on_dataset(model_path, squad['validation'])
# split_data_by_heuristics finds the best threshold for selected heuristic, and segment the dataset by this threshold
threshold_distance_dictionary, dataset = measurer.find_longest_distance(dataset, bias_id)
best_heuristic_threshold, bias_significance, distances_dict = threshold_distance_dictionary

print("Model '%s' bias '%s' significance is %s" % (model_path, bias_id, bias_significance))
```

2. Split data according to the selected heuristic to most 
significantly differentiate the model's performance ([full example](isbiased_examples/split_data_by_heuristic_example.py)):
```python

# dataset from find_longest_distance() contain the flags of the dataset segmentation
biased_dataset, nonbiased_dataset = measurer.split_data_by_heuristics(dataset, squad['train'], bias_id)
# now, biasedDataset contain biased (=better-performing samples) and unbiasedDataset contain unbiased (=worse-performing) data
# segments can also be used e.g. to train a "bias model", 
# or to manually evaluate a difference in the model's performance on unbiased segment
```

## Considerations

Note that the intended use of _IsBiased_ library and Bias significance measure has the following limitations:
* Bias significance provides a complement to OOD evaluation of models, and should not be used in standalone, as the bias significance can also be reduced by merely worsening model's quality on unbiased data subset.
* The value of Bias significance comprises merely a lower bound of the model's maximum performance polarisation by the exploited bias, as we have no guarantees on the optimality of the found threshold. We say that 'the model's bias is at least of the Bias significance value'. If you compare two different models, you should look for an optimal threshold of each bias for each model, unless you are interested in a specific bias configuration.

## Other parts of this repository
* See the [isbiased_examples](isbiased_examples) directory contain other examples of using _IsBiased_ library.
* See the [debiasing_methods](debiasing_methods) directory if you wish to train new debiasing models.
These models [will be available here after anonymity period].
* The [visualization](visualization) directory contain raw data from our measurements that we used in creating figures of the corresponding article.
