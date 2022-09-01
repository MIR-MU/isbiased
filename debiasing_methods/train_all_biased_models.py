from adaptor.adapter import Adapter
from adaptor.evaluators.question_answering import F1ScoreForQA
from adaptor.lang_module import LangModule
from adaptor.objectives.question_answering import ExtractiveQA
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from isbiased.bias_significance import BiasSignificanceMeasure

# parameters
starting_checkpoint = "bert-base-cased"
# full_dataset_model_path = "../models/electra-base-discriminator-finetuned-squad_with_callbacks_baseline"
biased_model_path = "../models/bert-base-cased"

num_val_samples = 200

trained_biases = ['kth_sentence', 'cosine_similarity', 'answer_length', 'max_sim_ents', 'answer_subject_positions']
# end parameters

squad_en = load_dataset("squad")
measurer = BiasSignificanceMeasure()

metrics, dataset = measurer.evaluate_model_on_dataset(biased_model_path, squad_en['train'])

for bias_id in trained_biases:
    lang_module = LangModule(starting_checkpoint)

    val_metrics = [F1ScoreForQA()]

    biasedDataset, unbiasedDataset = measurer.split_data_by_heuristics(dataset, squad_en['train'], bias_id)

    squad_train_biased = biasedDataset.filter(lambda entry: len(entry["context"]) < 2000)

    biased_objective = ExtractiveQA(lang_module,
                                    texts_or_path=squad_train_biased["question"],
                                    text_pair_or_path=squad_train_biased["context"],
                                    labels_or_path=[a["text"][0] for a in squad_train_biased["answers"]],
                                    val_texts_or_path=squad_en["validation"]["question"][:num_val_samples],
                                    val_text_pair_or_path=squad_en["validation"]["context"][:num_val_samples],
                                    val_labels_or_path=[a["text"][0] for a in squad_en["validation"]["answers"]][:num_val_samples],
                                    batch_size=30,
                                    val_evaluators=val_metrics,
                                    objective_id="SQUAD-en-biased")

    non_biased_objective = ExtractiveQA(lang_module,
                                        texts_or_path=unbiasedDataset["question"],
                                        text_pair_or_path=unbiasedDataset["context"],
                                        labels_or_path=[a["text"][0] for a in unbiasedDataset["answers"]],
                                        val_texts_or_path=unbiasedDataset["question"][:num_val_samples],
                                        val_text_pair_or_path=unbiasedDataset["context"][:num_val_samples],
                                        val_labels_or_path=[a["text"][0] for a in unbiasedDataset["answers"]][:num_val_samples],
                                        batch_size=3,
                                        val_evaluators=[F1ScoreForQA()],
                                        objective_id="SQUAD-en-non-biased",
                                        share_other_objective_head=biased_objective)

    training_arguments = AdaptationArguments(output_dir="biased-%s-%s" % (bias_id, starting_checkpoint),
                                             learning_rate=4e-5,
                                             stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                             do_train=True,
                                             do_eval=True,
                                             warmup_steps=1000,
                                             max_steps=100000,
                                             # gradient_accumulation_steps=5,
                                             logging_steps=100,
                                             save_steps=1000,
                                             eval_steps=1000,
                                             num_train_epochs=100,
                                             evaluation_strategy="steps",
                                             stopping_patience=10,
                                             save_total_limit=20)
    schedule = ParallelSchedule(objectives=[biased_objective],
                                extra_eval_objectives=[non_biased_objective],
                                args=training_arguments)

    adapter = Adapter(lang_module, schedule, args=training_arguments)
    adapter.train()
