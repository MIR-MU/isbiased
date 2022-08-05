import math
from typing import Optional, List, Union, Callable, Dict, Tuple

import torch
from datasets import Dataset
from torch import nn
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    TrainerCallback
from transformers.trainer_utils import PredictionOutput, speed_metrics, EvalPrediction


class DistillerTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            teacher_predictions : Optional[Dataset] = None,
            biased_predictions : Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super(Trainer, self).__init__(model=model,
                                      args=args,
                                      data_collator=data_collator,
                                      train_dataset=train_dataset,
                                      eval_dataset=eval_dataset,
                                      tokenizer=tokenizer,
                                      model_init=model_init,
                                      compute_metrics=compute_metrics,
                                      callbacks=callbacks,
                                      optimizers=optimizers,
                                      preprocess_logits_for_metrics=preprocess_logits_for_metrics, )

        self.teacher_predictions = teacher_predictions
        self.biased_predictions = biased_predictions
