import logging
from typing import Optional, Union, Dict, Tuple

import torch
from adaptor.objectives.question_answering import ExtractiveQA
from torch.nn import functional as F
from transformers import BatchEncoding
from transformers.modeling_outputs import QuestionAnsweringModelOutput

logger = logging.getLogger()


class LearnedMixinH(ExtractiveQA):
    # https://github.com/chrisc36/debias/blob/master/debias/modules/qa_debias_loss_functions.py#L26
    # https://github.com/chrisc36/debias/blob/master/debias/bert/clf_debias_loss_functions.py#L48

    def __init__(self, *args, biased_model: 'QuestionAnsweringModel', device: str, penalty: float = 0.03, **kwargs):
        super().__init__(*args, **kwargs)
        logger.warning("We have no way to check that given biased_model is really for QA :/ Be sure to pass QA model.")
        self.compatible_head_model.config.output_hidden_states = True  # we fit bias scalar on the model hidden states

        # devices of all modules are to be set according to the compatible_head_model's device
        self.biased_model = biased_model.to(device)
        self.penalty = penalty

        self.learned_bias_scalar = torch.nn.Linear(in_features=2 * self.compatible_head_model.config.hidden_size,
                                                   out_features=1,
                                                   device=device)

    def _bias_prediction(self,
                         inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]]
                         ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        model_outputs = self.biased_model(**{k: t for k, t in inputs.items()
                                             if k in ("input_ids", "attention_mask", "token_type_ids")})

        return model_outputs.start_logits, model_outputs.end_logits

    def _compute_loss(self,
                      model_outputs: QuestionAnsweringModelOutput,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:

        biased_start_logits, biased_end_logits = self._bias_prediction(inputs)

        start_lprobs = model_outputs.start_logits.log_softmax(1)
        end_lprobs = model_outputs.end_logits.log_softmax(1)

        start_hidden = model_outputs.hidden_states[-1][torch.arange(0, labels.shape[0]), start_lprobs.argmax(-1)]
        end_hidden = model_outputs.hidden_states[-1][torch.arange(0, labels.shape[0]), end_lprobs.argmax(-1)]

        bias_scale = self.learned_bias_scalar(torch.hstack([start_hidden, end_hidden]))
        bias_scale = F.softplus(bias_scale)

        biased_start_logits = biased_start_logits * bias_scale
        biased_end_logits = biased_end_logits * bias_scale

        biased_start_lprobs = biased_start_logits.log_softmax(1)
        biased_end_lprobs = biased_end_logits.log_softmax(1)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # TODO check: summing up trained logPROBS with bias LOGITS! Different scaling, but consistent with original git.
        # though inconsistent with the paper (https://aclanthology.org/D19-1418.pdf Sec. 3.2.4)
        start_loss = cross_entropy_loss(start_lprobs + biased_start_logits, inputs["start_position"])
        end_loss = cross_entropy_loss(end_lprobs + biased_end_logits, inputs["end_position"])

        total_loss = (start_loss + end_loss) / 2

        # entropy penalty (https://aclanthology.org/D19-1418.pdf Section 3.2.5)
        start_entropy = -(torch.exp(biased_start_lprobs) * biased_start_lprobs).sum(1).mean(0)
        end_entropy = -(torch.exp(biased_end_lprobs) * biased_end_lprobs).sum(1).mean(0)

        return total_loss + ((self.penalty / 2) * (start_entropy + end_entropy))
