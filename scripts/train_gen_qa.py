from typing import List, Iterable, Union, Dict

import torch
from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU
from adaptor.lang_module import LangModule
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from datasets import load_dataset
from transformers import BatchEncoding

training_arguments = AdaptationArguments(output_dir="train_dir",
                                         learning_rate=2e-5,
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=30,
                                         eval_steps=2,
                                         logging_steps=10,
                                         save_steps=1000,
                                         num_train_epochs=50,
                                         evaluation_strategy="steps",
                                         save_total_limit=10,
                                         stopping_patience=30)
eval_examples = 200

# priming
num_demonstrations = 3

val_metrics = [BLEU(**{"additional_sep_char": "▁"}, decides_convergence=True)]


def _construct_priming_prompt(previous_examples: List[str], current_example: str) -> str:
    return " ".join(previous_examples + [current_example])


# lang_module = LangModule("Helsinki-NLP/opus-mt-en-uk")  # TODO set for debugging
lang_module = LangModule("t5-large")

# priming
per_type_examples = {}

qa_en = load_dataset("squad")
qa_train = qa_en["train"].filter(lambda entry: len(entry["context"]) < 2000)


def _get_en_squad_categories(data) -> List[str]:
    return [question.split()[0] if not question.startswith("To")
            else " ".join(question.split()[:2])
            for question in data["question"]]


class GenQAObj(Sequence2Sequence):

    @staticmethod
    def _input_from_question_context(question: str, context: str) -> str:
        return "%s %s" % (context, question)

    def _get_inputs_iterator(self, split: str) -> Iterable[Union[BatchEncoding, Dict[str, torch.Tensor]]]:
        questions_iter, contexts_iter, answers_iter = self._per_split_iterators(split)

        input_texts_iter = (self._input_from_question_context(q, c) for q, c in zip(questions_iter, contexts_iter))

        collated_iter = self._get_seq2seq_collated_iterator(input_texts_iter, answers_iter)

        return collated_iter


train_qa = GenQAObj(lang_module,
                    texts_or_path=qa_train["question"],
                    text_pair_or_path=qa_train["context"],
                    val_texts_or_path=qa_en["validation"]["question"][-eval_examples:],
                    val_text_pair_or_path=qa_en["validation"]["context"][-eval_examples:],
                    labels_or_path=[a["text"][0] for a in qa_train["answers"]],
                    val_labels_or_path=[a["text"][0] for a in qa_en["validation"]["answers"]][-eval_examples:],
                    batch_size=1,
                    val_evaluators=val_metrics,
                    source_lang_id="en",
                    objective_id="SQuAD-en")

schedule = ParallelSchedule(objectives=[train_qa],
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)
adapter.train()
