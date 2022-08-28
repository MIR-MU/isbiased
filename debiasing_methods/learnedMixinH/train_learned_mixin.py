import argparse

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

parser = argparse.ArgumentParser()

parser.add_argument("--trained_model", default="bert-base-cased", type=str, help="Model to be debiased")
parser.add_argument("--full_model_path", type=str, help="A model trained on a full dataset, "
                                                        "presumably having biased predictions")
parser.add_argument("--biased_model_path", type=str, help="Bias model")
parser.add_argument("--bias_id", type=str, help="On which bias to train model. Supports all biases of 'isbiased' lib. "
                                             "Possible values: 'similar_words','distances','kth_sentence',"
                                             "'cosine_similarity', 'answer_length',"
                                             "'max_sim_ents','answer_subject_positions'")
parser.add_argument("--num_val_samples", type=int, help="A number of validation samples", default=200)

args = parser.parse_args()

lang_module = LangModule(args.trained_model)

device = "cuda" if torch.cuda.is_available() else "cpu"
biased_model = AutoModelForQuestionAnswering.from_pretrained(args.biased_model_path).to(device)

# dataset resolution
squad_en = load_dataset("squad")
squad_train = squad_en["train"].filter(lambda entry: len(entry["context"]) < 2000)

mixin_objective = LearnedMixinH(lang_module,
                                biased_model=biased_model,
                                device="cuda",
                                texts_or_path=squad_train["question"],
                                text_pair_or_path=squad_train["context"],
                                labels_or_path=[a["text"][0] for a in squad_train["answers"]],
                                val_texts_or_path=squad_en["validation"]["question"][:args.num_val_samples],
                                val_text_pair_or_path=squad_en["validation"]["context"][:args.num_val_samples],
                                val_labels_or_path=[a["text"][0] for a in squad_en["validation"]["answers"]][
                                                   :args.num_val_samples],
                                batch_size=30,
                                val_evaluators=[F1ScoreForQA()],
                                objective_id="SQUAD-en",
                                penalty=0.4,
                                bias_scale_proportion=10)

# bias logging:
measurer = BiasSignificanceMeasure()

# we need already-trained model for splitting the data to biased+non-biased
# this finds the optimal threshold for the heuristic and adds a splitting attribute to the dataset
metrics, dataset = measurer.evaluate_model_on_dataset(args.full_model_path, squad_en['validation'])

# segments the dataset by pre-computed threshold
biasedDataset, unbiasedDataset = measurer.split_data_by_heuristics(dataset, squad_en["train"], args.bias_id)

biased_objective = ExtractiveQA(lang_module,
                                texts_or_path=biasedDataset["question"],
                                text_pair_or_path=biasedDataset["context"],
                                labels_or_path=[a["text"][0] for a in biasedDataset["answers"]],
                                val_texts_or_path=biasedDataset["question"][:args.num_val_samples],
                                val_text_pair_or_path=biasedDataset["context"][:args.num_val_samples],
                                val_labels_or_path=[a["text"][0] for a in biasedDataset["answers"]][:args.num_val_samples],
                                batch_size=30,
                                val_evaluators=[F1ScoreForQA()],
                                objective_id="SQUAD-en-biased",
                                share_other_objective_head=mixin_objective)

non_biased_objective = ExtractiveQA(lang_module,
                                    texts_or_path=unbiasedDataset["question"],
                                    text_pair_or_path=unbiasedDataset["context"],
                                    labels_or_path=[a["text"][0] for a in unbiasedDataset["answers"]],
                                    val_texts_or_path=unbiasedDataset["question"][:args.num_val_samples],
                                    val_text_pair_or_path=unbiasedDataset["context"][:args.num_val_samples],
                                    val_labels_or_path=[a["text"][0] for a in unbiasedDataset["answers"]][
                                                       :args.num_val_samples],
                                    batch_size=30,
                                    val_evaluators=[F1ScoreForQA()],
                                    objective_id="SQUAD-en-non-biased",
                                    share_other_objective_head=mixin_objective)
# end: bias logging

training_arguments = AdaptationArguments(output_dir="LMix-%s-%s-checkpoints" % (args.bias_id, args.trained_model),
                                         learning_rate=5e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=2000,
                                         max_steps=200000,
                                         eval_steps=1000,
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

print("This was LearnedMixin training run with the following args: %s " % str(args))
