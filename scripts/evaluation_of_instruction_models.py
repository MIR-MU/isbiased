from bias_significance import *
from datasets import load_dataset
import sys

def squad_eval_find_distance(model_path):
    squad = load_dataset("squad")

    measurer = BiasSignificanceMeasure()

    metrics, dataset = measurer.evaluate_instruction_model(model_path, squad['validation'])

    heuristics = ['similar_words', 'distances', 'kth_sentence', 'cosine_similarity', 'answer_length', 'max_sim_ents', 'answer_subject_positions']

    for h in heuristics:
        zusammen, dataset_ret = measurer.find_longest_distance(model_path, dataset, h)
        best_threshold, max_distance, distances_dictionary = zusammen
    
        with open("./bias_significance_and_squad.csv", "a") as file:
            file.write(f"\n{model_path},{heuristic},{metrics['exact_match']},{metrics['f1']},{best_threshold},{max_distance},{distances_dictionary}")


if __name__ == '__main__':
    model_paths = sys.argv[1].split(',')

    for model_path in model_paths:
        squad_eval_find_distance(model_path)