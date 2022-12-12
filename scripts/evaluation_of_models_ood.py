from isbiased.bias_significance import BiasSignificanceMeasure
from datasets import load_dataset, Dataset
import pandas as pd
import sys

def evaluate_ood_datasets(model_path):
    measurer = BiasSignificanceMeasure()

    eval_dataset = load_dataset("adversarial_qa", "adversarialQA")
    metrics, dataset = measurer.evaluate_model_on_dataset(model_path, eval_dataset['validation'])

    with open("./performance_on_ood_datasets.csv", "a") as file:
        file.write(f"\n{model_path},adversarialqa,{metrics['exact_match']},{metrics['f1']}")

    df = pd.read_json('./ood_datasets/nq_dev_formatted.json')
    eval_dataset = Dataset.from_pandas(df)
    metrics, dataset = measurer.evaluate_model_on_dataset(model_path, eval_dataset)

    with open("./performance_on_ood_datasets.csv", "a") as file:
        file.write(f"\n{model_path},nq,{metrics['exact_match']},{metrics['f1']}")

    df = pd.read_json('./ood_datasets/triviaqa_dev_formatted.json')
    eval_dataset = Dataset.from_pandas(df)
    metrics, dataset = measurer.evaluate_model_on_dataset(model_path, eval_dataset)

    with open("./performance_on_ood_datasets.csv", "a") as file:
        file.write(f"\n{model_path},triviaqa,{metrics['exact_match']},{metrics['f1']}")


if __name__ == '__main__':
    model_paths = sys.argv[1].split(',')

    for model_path in model_paths:
        evaluate_ood_datasets(model_path)
