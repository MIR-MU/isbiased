# TODO: pip install git+https://github.com/gaussalgo/adaptor.git@fix/early_stopping

from adaptor.adapter import Adapter
from adaptor.evaluators.question_answering import F1ScoreForQA
from adaptor.lang_module import LangModule
from adaptor.objectives.question_answering import ExtractiveQA
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

from isbiased.bias_significance import BiasSignificanceMeasure

# changeable parameters
biased_model_id = "bert-base-cased"
# local:
# biased_model_id = "bert-base-multilingual-cased"

bias_id = "distances"

# fixed env-specific parameters
full_dataset_model_path = "/mnt/local/disk1/klasifikace_reflexe/think_twice/isbiased/models/roberta-base-orig"
# full_dataset_model_path = "../models/electra-base-discriminator-finetuned-squad_with_callbacks_baseline"

num_val_samples = 200

# validation and biased training data
squad_en = load_dataset("squad")

# measurer = BiasSignificanceMeasure(squad_en['train'].select(range(2000)))
measurer = BiasSignificanceMeasure(squad_en['train'])
# we need already-trained model for this
# measurer.evaluate_model_on_dataset(full_dataset_model_path, squad_en['validation'].select(range(2000)))
metrics, dataset = measurer.evaluate_model_on_dataset(full_dataset_model_path, squad_en['train'])

# end: parameters

biasedDataset, unbiasedDataset = measurer.split_data_by_heuristics(dataset, squad_en['train'], bias_id)

squad_train_biased = biasedDataset.filter(lambda entry: len(entry["context"]) < 2000)


lang_module = LangModule(biased_model_id)

training_arguments = AdaptationArguments(output_dir="train_dir-%s-%s" % (bias_id, biased_model_id),
                                         learning_rate=4e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_NUM_STEPS,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=5,
                                         eval_steps=200,
                                         logging_steps=100,
                                         save_steps=1000,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps",
                                         stopping_patience=200)

val_metrics = [F1ScoreForQA(decides_convergence=True)]

mixin_objective = ExtractiveQA(lang_module,
                               texts_or_path=squad_train_biased["question"],
                               text_pair_or_path=squad_train_biased["context"],
                               labels_or_path=[a["text"][0] for a in squad_train_biased["answers"]],
                               val_texts_or_path=squad_en["validation"]["question"][:num_val_samples],
                               val_text_pair_or_path=squad_en["validation"]["context"][:num_val_samples],
                               val_labels_or_path=[a["text"][0] for a in squad_en["validation"]["answers"]][:num_val_samples],
                               batch_size=3,
                               val_evaluators=val_metrics,
                               objective_id="SQUAD-en-biased")

biased_objective = ExtractiveQA(lang_module,
                                texts_or_path=biasedDataset["question"],
                                text_pair_or_path=biasedDataset["context"],
                                labels_or_path=[a["text"][0] for a in biasedDataset["answers"]],
                                val_texts_or_path=biasedDataset["question"][:num_val_samples],
                                val_text_pair_or_path=biasedDataset["context"][:num_val_samples],
                                val_labels_or_path=[a["text"][0] for a in biasedDataset["answers"]][:num_val_samples],
                                batch_size=3,
                                val_evaluators=[F1ScoreForQA()],
                                objective_id="SQUAD-en-biased",
                                share_other_objective_head=mixin_objective)

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
                                    share_other_objective_head=mixin_objective)

schedule = ParallelSchedule(objectives=[mixin_objective],
                            extra_eval_objectives=[biased_objective, non_biased_objective],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
