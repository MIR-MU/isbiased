from adaptor.adapter import Adapter
from adaptor.evaluators.question_answering import F1ScoreForQA
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering

from learnedMixinH.objective import LearnedMixinH

model_name = "bert-base-multilingual-cased"
biased_model_path = "bert-base-multilingual-cased"

lang_module = LangModule(model_name)

biased_model = AutoModelForQuestionAnswering.from_pretrained(biased_model_path)

training_arguments = AdaptationArguments(output_dir="checkpoint_dir",
                                         learning_rate=4e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=5,
                                         eval_steps=200,
                                         logging_steps=100,
                                         save_steps=1000,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps")

val_metrics = [F1ScoreForQA(decides_convergence=True)]

# english SQuAD
squad_en = load_dataset("squad")
squad_train = squad_en["train"].filter(lambda entry: len(entry["context"]) < 2000)

mixin_objective = LearnedMixinH(lang_module,
                                biased_model=biased_model,
                                texts_or_path=squad_train["question"],
                                text_pair_or_path=squad_train["context"],
                                labels_or_path=[a["text"][0] for a in squad_train["answers"]],
                                val_texts_or_path=squad_en["validation"]["question"][:200],
                                val_text_pair_or_path=squad_en["validation"]["context"][:200],
                                val_labels_or_path=[a["text"][0] for a in squad_en["validation"]["answers"]][:200],
                                batch_size=3,
                                val_evaluators=val_metrics,
                                objective_id="SQUAD-en")

schedule = ParallelSchedule(objectives=[mixin_objective],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
