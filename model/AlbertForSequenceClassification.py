from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(
            0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output+0.1)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss(reduction='sum')
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
