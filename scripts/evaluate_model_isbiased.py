from datasets import load_dataset

from isbiased.bias_significance import BiasSignificanceMeasure

# parameters:
model_path = "debiasing_methods/checkpoint_dir/checkpoint-29000/SQUAD-en-LearnedMixinH"
model_path = "../models/bert-base-orig"
# model_path = 'csarron/bert-base-uncased-squad-v1'  # from SQuAD-QA model from HF

split = "validation"
# dataset = load_dataset("squad")[split]
dataset = load_dataset("adversarial_qa", "adversarialQA")[split]
# end parameters

measurer = BiasSignificanceMeasure()

metrics, dataset = measurer.evaluate_model_on_dataset(model_path, dataset.select(range(1000)), batch_size=1)
print("Checkpoint %s performance: %s" % (model_path, metrics))
