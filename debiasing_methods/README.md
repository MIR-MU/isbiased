# Debiasing training experiments

You can run and retrain all our models by `cd debiasing_methods && export PYTHONPATH=$pwd:{$PYTHONPATH}`, using the following commands

0. Bias models: ```python learnedMixinH/train_all_biased_models.py```
This will produce checkpoints from which we pick the one, where the logged performance (F1 score)
of "bias" and "non-bias" model differs the most
1. LearnedMixin: ```python learnedMixinH/train_lmix.py```
2. Confidence Regularization: In addition, requires a teacher (i.e. conventional extractive QA) 
model, that you can obtain from ```python confidenceRegularization/train_and_predict_parent.py```. 
Then, you can train CReg using: ```python confidenceRegularization/train_creg.py```.

In the case of both debiasing methods, the training produces the checkpoints, 
from which we pick the one with the highest absolute in-distribution perfornace (F1 score).

The scripts (1. and 2.) expect as input the bias model, or in case of Confidence Regularization,
also the teacher model. See their CLI help for specifics, e.g: ```python learnedMixinH/train_lmix.py -h```
