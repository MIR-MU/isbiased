import torch.cuda
from adaptor.adapter import Adapter
from adaptor.evaluators.question_answering import F1ScoreForQA
from adaptor.lang_module import LangModule
from adaptor.objectives.question_answering import ExtractiveQA
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering

from isbiased.bias_significance import BiasSignificanceMeasure
from learnedMixinH.objective import LearnedMixinH

# parameters
bias_id = "distances"

model_name = "bert-base-multilingual-cased"
biased_model_path = "../models/biased/%s/bert-base" % bias_id

num_val_samples = 200
# end parameters

lang_module = LangModule(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
biased_model = AutoModelForQuestionAnswering.from_pretrained(biased_model_path).to(device)

# dataset resolution
squad_en = load_dataset("squad")
squad_train = squad_en["train"].filter(lambda entry: len(entry["context"]) < 2000)


mixin_objective = LearnedMixinH(lang_module,
                                biased_model=biased_model,
                                device="cuda",
                                texts_or_path=squad_train["question"],
                                text_pair_or_path=squad_train["context"],
                                labels_or_path=[a["text"][0] for a in squad_train["answers"]],
                                val_texts_or_path=squad_en["validation"]["question"][:num_val_samples],
                                val_text_pair_or_path=squad_en["validation"]["context"][:num_val_samples],
                                val_labels_or_path=[a["text"][0] for a in squad_en["validation"]["answers"]][:num_val_samples],
                                batch_size=3,
                                val_evaluators=[F1ScoreForQA()],
                                objective_id="SQUAD-en")

# bias logging:
full_dataset_model_path = "/mnt/local/disk1/klasifikace_reflexe/think_twice/isbiased/models/roberta-base-orig"
# full_dataset_model_path = "../models/electra-base-discriminator-finetuned-squad_with_callbacks_baseline"

# measurer = BiasSignificanceMeasure(squad_en['train'].select(range(2000)))
measurer = BiasSignificanceMeasure(squad_en['train'])
# we need already-trained model for this
# measurer.evaluate_model_on_dataset(full_dataset_model_path, squad_en['validation'].select(range(2000)))
metrics, dataset = measurer.evaluate_model_on_dataset(full_dataset_model_path, squad_en['train'])

biasedDataset, unbiasedDataset = measurer.split_data_by_heuristics(dataset, squad_en['train'], bias_id)

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
# end: bias logging

training_arguments = AdaptationArguments(output_dir="checkpoint_dir",
                                         learning_rate=2e-4,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=5,
                                         eval_steps=100,
                                         logging_steps=100,
                                         save_steps=1000,
                                         # stopping_patience=100,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps")

schedule = ParallelSchedule(objectives=[mixin_objective],
                            extra_eval_objectives=[biased_objective, non_biased_objective],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
