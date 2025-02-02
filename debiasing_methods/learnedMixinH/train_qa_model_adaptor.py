# TODO: pip install git+https://github.com/gaussalgo/adaptor.git@fix/early_stopping
# TODO: pip install datasets==3.2.0

from adaptor.adapter import Adapter
from adaptor.evaluators.question_answering import F1ScoreForQA
from adaptor.lang_module import LangModule
from adaptor.objectives.question_answering import ExtractiveQA
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset

# changeable parameters
base_model_id = "bert-base-cased"
# local:
# biased_model_id = "bert-base-multilingual-cased"


num_val_samples = 200

# validation and biased training data
subset = "disfl_qa"

ds = load_dataset("kowndinya23/bigbench_zero_shot", subset)
# end: parameters


ds = ds.filter(lambda entry: len(entry["inputs"]) < 2000)

lang_module = LangModule(base_model_id)

training_arguments = AdaptationArguments(output_dir="train_dir-%s-%s" % (base_model_id, subset),
                                         learning_rate=4e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_NUM_STEPS,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=3,
                                         save_total_limit=6,
                                         eval_steps=500,
                                         logging_steps=100,
                                         save_steps=500,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps",
                                         stopping_patience=5,
                                         no_cuda=True)

val_metrics = [F1ScoreForQA(decides_convergence=True)]

qa_objective = ExtractiveQA(lang_module,
                            texts_or_path=ds["train"]["inputs"],
                            text_pair_or_path=["" for _ in ds["train"]["inputs"]],
                            labels_or_path=[a[0] for a in ds["train"]["targets"]],
                            val_texts_or_path=ds["validation"]["inputs"][:num_val_samples],
                            val_text_pair_or_path=["" for _ in ds["validation"]["inputs"]][:num_val_samples],
                            val_labels_or_path=[a[0] for a in ds["validation"]["targets"]][:num_val_samples],
                            batch_size=8,
                            val_evaluators=val_metrics,
                            objective_id=subset)

schedule = ParallelSchedule(objectives=[qa_objective],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
