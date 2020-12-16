import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertModel,
    AlbertModel,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    AlbertPreTrainedModel,
    BertPreTrainedModel
)


class BertNCPQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNCPQA, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class AlbertNCPQA(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertNCPQA, self).__init__(config)

        self.mrc = AlbertForQuestionAnswering(config)
        self.content = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None):

        outputs = self.mrc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )


        if start_positions is not None and end_positions is not None:
            boundary_loss = outputs[0]
            start_logits, end_logits = outputs[1], outputs[2]

        return outputs  # (loss), start_logits, end_logits, ( ), (attentions)


class Content(nn.Module):
    def __init__(self, config):
        super(Content, self).__init__(config)
        self.content = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, sequence_output=None):

        pc = self.content(sequence_output)