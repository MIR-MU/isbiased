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
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):

        super().__init__(
            model= model,
            args= args,
            data_collator= data_collator,
            train_dataset= train_dataset,
            eval_dataset= eval_dataset,
            tokenizer= tokenizer,
            model_init= model_init,
            compute_metrics= compute_metrics,
            callbacks= callbacks,
            optimizers= optimizers,
            preprocess_logits_for_metrics= preprocess_logits_for_metrics,)

        # super(Trainer, self).__init__(self,model=model,
        #                               args=args,
        #                               data_collator=data_collator,
        #                               train_dataset=train_dataset,
        #                               eval_dataset=eval_dataset,
        #                               tokenizer=tokenizer,
        #                               model_init=model_init,
        #                               compute_metrics=compute_metrics,
        #                               callbacks=callbacks,
        #                               optimizers=optimizers,
        #                               preprocess_logits_for_metrics=preprocess_logits_for_metrics, )




    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
