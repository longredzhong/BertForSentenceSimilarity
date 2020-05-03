from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel,AlbertEmbeddings
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from model.utils.ESIM_layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from model.utils.utils import get_mask, replace_masked
import torch

class ESIM(AlbertPreTrainedModel):
    def __init__(self, config):
        super(ESIM, self).__init__(config)
        self.num_labels = config.num_labels
        self.dropout = config.dropout
        self._word_embedding = AlbertEmbeddings(config)
        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        config.embedding_size,
                                        config.hidden_size,
                                        bidirectional=True)
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=config.dropout)
        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4*2*config.hidden_size,
                                                   config.hidden_size),
                                         nn.ReLU())
        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           config.hidden_size,
                                           config.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=config.dropout),
                                             nn.Linear(2*4*config.hidden_size,
                                                       config.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=config.dropout),
                                             nn.Linear(config.hidden_size,
                                                       config.num_labels))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                input_1,
                input_1_lengths,
                input_2,
                input_2_lengths,
                labels):
        """
        Args:
            input_1: A batch of varaible length sequences of word indices
                representing input_1. The batch is assumed to be of size
                (batch, input_1_length).
            input_1_lengths: A 1D tensor containing the lengths of the
                input_1 in 'input_1'.
            input_2: A batch of varaible length sequences of word indices
                representing input_2. The batch is assumed to be of size
                (batch, input_2_length).
            input_2_lengths: A 1D tensor containing the lengths of the
                input_2 in 'input_2'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        input_1_mask = get_mask(input_1, input_1_lengths).to(self.device)
        input_2_mask = get_mask(input_2, input_2_lengths)\
            .to(self.device)

        embedded_input_1 = self._word_embedding(input_1)
        embedded_input_2 = self._word_embedding(input_2)

        if self.dropout:
            embedded_input_1 = self._rnn_dropout(embedded_input_1)
            embedded_input_2 = self._rnn_dropout(embedded_input_2)

        encoded_input_1 = self._encoding(embedded_input_1,
                                          input_1_lengths)
        encoded_input_2 = self._encoding(embedded_input_2,
                                            input_2_lengths)

        attended_input_1, attended_input_2 =\
            self._attention(encoded_input_1, input_1_mask,
                            encoded_input_2, input_2_mask)

        enhanced_input_1 = torch.cat([encoded_input_1,
                                       attended_input_1,
                                       encoded_input_1 - attended_input_1,
                                       encoded_input_1 * attended_input_1],
                                      dim=-1)
        enhanced_input_2 = torch.cat([encoded_input_2,
                                         attended_input_2,
                                         encoded_input_2 -
                                         attended_input_2,
                                         encoded_input_2 *
                                         attended_input_2],
                                        dim=-1)

        projected_input_1 = self._projection(enhanced_input_1)
        projected_input_2 = self._projection(enhanced_input_2)

        if self.dropout:
            projected_input_1 = self._rnn_dropout(projected_input_1)
            projected_input_2 = self._rnn_dropout(projected_input_2)

        v_ai = self._composition(projected_input_1, input_1_lengths)
        v_bj = self._composition(projected_input_2, input_2_lengths)

        v_a_avg = torch.sum(v_ai * input_1_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(input_1_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * input_2_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(input_2_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, input_1_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, input_2_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        # probabilities = nn.functional.softmax(logits, dim=-1)
        outputs = (logits,)
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

        return outputs

def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
