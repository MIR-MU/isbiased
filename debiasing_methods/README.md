# Training debiased models

Before running the training scripts, note that the debiased models from our experiments [will be available here after anonymity period].

1. Run scripts from this directory: `cd debiasing_methods && export PYTHONPATH="${PYTHONPATH}:$(pwd)"`
2. Install requirements of selected debiasing method: `pip install -r requirements.txt`
3. All debiasing methods require bias models, which can be downloaded, or trained as follows:
```shell
CUDA_VISIBLE_DEVICES=XX python train_biased_model.py
```
4. Confidence Regularization also requires teacher model, that can be trained as follows:
```shell
CUDA_VISIBLE_DEVICES=XX python train_standard_QA_model.py --model bert-base-cased
```
5. Train a selected debiasing method - note that you have to point the `biased_model_path` to the selected checkpoint of the bias model training. Further, the scripts contain some qualified defaults of `TrainingArguments`, that you might need to alter according to your environment (e.g. a size of GPU).
   1. LearnedMixin (addressing `max-sim-ents` bias):
   ```shell
   python learnedMixinH/train_learned_mixin.py --trained_model bert-base-cased \ 
                                               --full_model_path models/bert-base-cased \ 
                                               --biased_model_path models/biased/max_sim_ents \ 
                                               --bias_id max_sim_ents
   ```
   2. Confidence Regularization:
   ```shell
   python confidenceRegularization/train_distilled_model.py --trained_model bert-base-cased \
                                                            --biased_model models/biased/answer_length \ 
                                                            --teacher_model models/bert-base-cased \
                                                            --bias answer_length \
                                                            --eval_firstn 200
   ```
Note that the the specific checkpoints from each training run should be picked manually, according to the following criteria: 
1. to maximise the difference in F-scores for biased models, and
2. to maximise the overall F-score for the teacher model
3. to maximise the F-score on non-biased subset for the final models.
