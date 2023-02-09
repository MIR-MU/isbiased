from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from transformers import BertForQuestionAnswering, BertModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class LMixBertForQuestionAnswering(BertForQuestionAnswering):
    """
    Pre-trained Distilled Bert for Question Answering which uses LearnedMixinHLoss loss function

    Overides only to change loss function
    """

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, loss_fn: Callable):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fn = loss_fn

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            bias_probs_start=None,
            bias_probs_end=None,
            teacher_probs_start=None,
            teacher_probs_end=None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # TODO - check meaning of ignore_index
            # loss_fct = self.loss_fn(ignore_index=ignored_index)
            # fixme
            # bias and teacher_preds have 2 dimensions = 0: start, 1:end
            # start_loss = loss_fct(logits=start_logits, start_positions, bias_probs=bias_probs_start,
            # teacher_probs=teacher_probs_start, labels=labels)
            # TODO: resolve hidden states: look at the model output in debugger
            start_loss = self.loss_fn(logits=start_logits, bias_logits=bias_probs_start,
                                      hidden_states=outputs.last_hidden_state, labels=start_positions)
            end_loss = self.loss_fn(logits=end_logits, bias_logits=bias_probs_end,
                                    hidden_states=outputs.last_hidden_state, labels=end_positions)

            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
