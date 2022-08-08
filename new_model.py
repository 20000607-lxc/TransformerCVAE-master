from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.roberta.modeling_roberta import RobertaModel
from model import Generator, Discriminator, AverageSelfAttention
from transformers import BertModel
import copy
import random

def init_para_frompretrained_roberta(m, pm, share_para=True, decoder=False):

    m.embeddings.word_embeddings.weight = pm.embeddings.word_embeddings.weight
    m.embeddings.word_embeddings.weight.requires_grad = True
    m.embeddings.position_embeddings.weight = pm.embeddings.position_embeddings.weight
    m.embeddings.position_embeddings.weight.requires_grad = True
    m.embeddings.token_type_embeddings.weight = pm.embeddings.token_type_embeddings.weight
    m.embeddings.token_type_embeddings.weight.requires_grad = True
    m.embeddings.LayerNorm.weight = pm.embeddings.LayerNorm.weight
    m.embeddings.LayerNorm.weight.requires_grad = True
    m.embeddings.LayerNorm.bias = pm.embeddings.LayerNorm.bias
    m.embeddings.LayerNorm.bias.requires_grad = True

    for i in range(max(len(m.encoder.layer), len(pm.encoder.layer))): # RobertaLayer

        m.encoder.layer[i].attention.self.key.weight = pm.encoder.layer[i].attention.self.key.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.key.weight)
        m.encoder.layer[i].attention.self.key.weight.requires_grad = True
        m.encoder.layer[i].attention.self.key.bias = pm.encoder.layer[i].attention.self.key.bias if share_para else copy.copy(pm.encoder.layer[i].attention.self.key.bias)
        m.encoder.layer[i].attention.self.key.bias.requires_grad = True
        m.encoder.layer[i].attention.self.query.weight = pm.encoder.layer[i].attention.self.query.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.query.weight)
        m.encoder.layer[i].attention.self.query.weight.requires_grad = True
        m.encoder.layer[i].attention.self.query.bias = pm.encoder.layer[i].attention.self.query.bias if share_para else copy.copy(pm.encoder.layer[i].attention.self.query.bias)
        m.encoder.layer[i].attention.self.query.bias.requires_grad = True
        m.encoder.layer[i].attention.self.value.weight = pm.encoder.layer[i].attention.self.value.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.value.weight)
        m.encoder.layer[i].attention.self.value.weight.requires_grad = True
        m.encoder.layer[i].attention.self.value.bias = pm.encoder.layer[i].attention.self.value.bias if share_para else copy.copy(pm.encoder.layer[i].attention.self.value.bias)
        m.encoder.layer[i].attention.self.value.bias.requires_grad = True
        # m.encoder.layer[i].attention.self.dropout = pm.encoder.layer[i].attention.self.dropout if share_para else copy.copy(pm.encoder.layer[i].attention.self.dropout)

        m.encoder.layer[i].attention.output.dense.weight = pm.encoder.layer[i].attention.output.dense.weight if share_para else copy.copy(pm.encoder.layer[i].attention.output.dense.weight)
        m.encoder.layer[i].attention.output.dense.weight.requires_grad = True
        m.encoder.layer[i].attention.output.dense.bias = pm.encoder.layer[i].attention.output.dense.bias if share_para else copy.copy(pm.encoder.layer[i].attention.output.dense.bias)
        m.encoder.layer[i].attention.output.dense.bias.requires_grad = True
        m.encoder.layer[i].attention.output.LayerNorm.weight = pm.encoder.layer[i].attention.output.LayerNorm.weight if share_para else copy.copy(pm.encoder.layer[i].attention.output.LayerNorm.weight)
        m.encoder.layer[i].attention.output.LayerNorm.weight.requires_grad = True
        m.encoder.layer[i].attention.output.LayerNorm.bias = pm.encoder.layer[i].attention.output.LayerNorm.bias if share_para else copy.copy(pm.encoder.layer[i].attention.output.LayerNorm.bias)
        m.encoder.layer[i].attention.output.LayerNorm.bias.requires_grad = True
        # m.encoder.layer[i].attention.output.dropout = pm.encoder.layer[i].attention.output.dropout if share_para else copy.copy(pm.encoder.layer[i].attention.output.dropout)

        m.encoder.layer[i].intermediate.dense.weight = pm.encoder.layer[i].intermediate.dense.weight if share_para else copy.copy(pm.encoder.layer[i].intermediate.dense.weight)
        m.encoder.layer[i].intermediate.dense.weight.requires_grad = True
        m.encoder.layer[i].intermediate.dense.bias = pm.encoder.layer[i].intermediate.dense.bias if share_para else copy.copy(pm.encoder.layer[i].intermediate.dense.bias)
        m.encoder.layer[i].intermediate.dense.bias.requires_grad = True
        m.encoder.layer[i].output.dense.weight = pm.encoder.layer[i].output.dense.weight if share_para else copy.copy(pm.encoder.layer[i].output.dense.weight)
        m.encoder.layer[i].output.dense.weight.requires_grad = True
        m.encoder.layer[i].output.dense.bias = pm.encoder.layer[i].output.dense.bias if share_para else copy.copy(pm.encoder.layer[i].output.dense.bias)
        m.encoder.layer[i].output.dense.bias.requires_grad = True
        m.encoder.layer[i].output.LayerNorm.weight = pm.encoder.layer[i].output.LayerNorm.weight if share_para else copy.copy(pm.encoder.layer[i].output.LayerNorm.weight)
        m.encoder.layer[i].output.LayerNorm.weight.requires_grad = True
        m.encoder.layer[i].output.LayerNorm.bias = pm.encoder.layer[i].output.LayerNorm.bias if share_para else copy.copy(pm.encoder.layer[i].output.LayerNorm.bias)
        m.encoder.layer[i].output.LayerNorm.bias.requires_grad = True
        # m.encoder.layer[i].output.dropout = pm.encoder.layer[i].output.dropout if share_para else copy.copy(pm.encoder.layer[i].output.dropout)

    if not decoder:
        m.pooler.dense.weight = pm.pooler.dense.weight
        m.pooler.dense.weight.requires_grad = True
        m.pooler.dense.bias = pm.pooler.dense.bias
        m.pooler.dense.bias.requires_grad = True
    print('finish sharing weight')


def init_para_frompretrained_deberta(m, pm, share_para=True):
    m.embeddings.word_embeddings.weight = pm.embeddings.word_embeddings.weight
    m.embeddings.word_embeddings.weight.requires_grad = True
    m.embeddings.LayerNorm.weight = pm.embeddings.LayerNorm.weight
    m.embeddings.LayerNorm.weight.requires_grad = True

    m.embeddings.LayerNorm.bias = pm.embeddings.LayerNorm.bias
    m.embeddings.LayerNorm.bias.requires_grad = True

    for i in range(max(len(m.encoder.layer), len(pm.encoder.layer))):
        m.encoder.layer[i].attention.self.in_proj.weight = pm.encoder.layer[i].attention.self.in_proj.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.in_proj.weight)
        m.encoder.layer[i].attention.self.in_proj.weight.requires_grad = True

        m.encoder.layer[i].attention.self.pos_proj.weight = pm.encoder.layer[i].attention.self.pos_proj.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.pos_proj.weight)
        m.encoder.layer[i].attention.self.pos_proj.weight.requires_grad = True

        m.encoder.layer[i].attention.self.pos_q_proj.weight = pm.encoder.layer[i].attention.self.pos_q_proj.weight if share_para else copy.copy(pm.encoder.layer[i].attention.self.pos_q_proj.weight)
        m.encoder.layer[i].attention.self.pos_q_proj.weight.requires_grad = True
        m.encoder.layer[i].attention.self.pos_q_proj.bias = pm.encoder.layer[i].attention.self.pos_q_proj.bias if share_para else copy.copy(pm.encoder.layer[i].attention.self.pos_q_proj.bias)
        m.encoder.layer[i].attention.self.pos_q_proj.bias.requires_grad = True

        m.encoder.layer[i].attention.output.dense.weight = pm.encoder.layer[i].attention.output.dense.weight if share_para else copy.copy(pm.encoder.layer[i].attention.output.dense.weight)
        m.encoder.layer[i].attention.output.dense.weight.requires_grad = True
        m.encoder.layer[i].attention.output.dense.bias = pm.encoder.layer[i].attention.output.dense.bias if share_para else copy.copy(pm.encoder.layer[i].attention.output.dense.bias)
        m.encoder.layer[i].attention.output.dense.bias.requires_grad = True
        m.encoder.layer[i].attention.output.LayerNorm.weight = pm.encoder.layer[i].attention.output.LayerNorm.weight if share_para else copy.copy(pm.encoder.layer[i].attention.output.LayerNorm.weight)
        m.encoder.layer[i].attention.output.LayerNorm.weight.requires_grad = True
        m.encoder.layer[i].attention.output.LayerNorm.bias = pm.encoder.layer[i].attention.output.LayerNorm.bias if share_para else copy.copy(pm.encoder.layer[i].attention.output.LayerNorm.bias)
        m.encoder.layer[i].attention.output.LayerNorm.bias.requires_grad = True

        m.encoder.layer[i].intermediate.dense.weight = pm.encoder.layer[i].intermediate.dense.weight if share_para else copy.copy(pm.encoder.layer[i].intermediate.dense.weight)
        m.encoder.layer[i].intermediate.dense.weight.requires_grad = True
        m.encoder.layer[i].intermediate.dense.bias = pm.encoder.layer[i].intermediate.dense.bias if share_para else copy.copy(pm.encoder.layer[i].intermediate.dense.bias)
        m.encoder.layer[i].intermediate.dense.bias.requires_grad = True
        m.encoder.layer[i].output.dense.weight = pm.encoder.layer[i].output.dense.weight if share_para else copy.copy(pm.encoder.layer[i].output.dense.weight)
        m.encoder.layer[i].output.dense.weight.requires_grad = True
        m.encoder.layer[i].output.dense.bias = pm.encoder.layer[i].output.dense.bias if share_para else copy.copy(pm.encoder.layer[i].output.dense.bias)
        m.encoder.layer[i].output.dense.bias.requires_grad = True
        m.encoder.layer[i].output.LayerNorm.weight = pm.encoder.layer[i].output.LayerNorm.weight if share_para else copy.copy(pm.encoder.layer[i].output.LayerNorm.weight)
        m.encoder.layer[i].output.LayerNorm.weight.requires_grad = True
        m.encoder.layer[i].output.LayerNorm.bias = pm.encoder.layer[i].output.LayerNorm.bias if share_para else copy.copy(pm.encoder.layer[i].output.LayerNorm.bias)
        m.encoder.layer[i].output.LayerNorm.bias.requires_grad = True

    m.encoder.rel_embeddings.weight = pm.encoder.rel_embeddings.weight
    m.encoder.rel_embeddings.weight.requires_grad = True
    print('finish sharing weight')

def share_weights_Conv(m , pm, share=True):
    m.bias = pm.bias
    m.weight = pm.weight

def share_weights_Attn(m, pm, share=True):
    m.attention_weights = pm.attention_weights

class MultiVAEModel(nn.Module):
    def __init__(self, encoder_model_class, decoder_model_class, encoder_name, decoder_name, device,  ed_share=False):
        super().__init__()
        self.device = device
        self.encoder_config = encoder_model_class[encoder_name][0].from_pretrained(encoder_name)
        self.decoder_config = decoder_model_class[decoder_name][0].from_pretrained(decoder_name)
        n_embd = 768

        self.mean = Conv1D(n_embd, n_embd)
        self.logvar = Conv1D(n_embd, n_embd)
        self.averageSelfAttention = AverageSelfAttention(n_embd, 'base')

        self.mean_prior = Conv1D(n_embd, n_embd)
        self.logvar_prior = Conv1D(n_embd, n_embd)
        self.averageSelfAttention_prior = AverageSelfAttention(n_embd, 'base')

        # if 'gpt' in decoder_name:
        share_weights_Conv(self.mean_prior, self.mean)
        share_weights_Conv(self.logvar_prior, self.logvar)
        share_weights_Attn(self.averageSelfAttention_prior, self.averageSelfAttention)

        self.input_proj = nn.Linear(n_embd, n_embd, bias=False)

        if decoder_name == 'gpt2':
            self.transformer = decoder_model_class[decoder_name][1].from_pretrained(decoder_name, config=self.decoder_config)
            self.lm_head = decoder_model_class[decoder_name][2].from_pretrained(decoder_name, config=self.decoder_config).lm_head
        else:
            self.decoder_config.is_decoder = True
            # notice add_pooling_layer=False for bert decoder
            self.transformer = decoder_model_class[decoder_name][1].from_pretrained(decoder_name, config=self.decoder_config, add_pooling_layer=False)
            self.lm_head = decoder_model_class[decoder_name][2](self.decoder_config)

            # self.transformer = BertLMHeadModel.from_pretrained("bert-base-cased", config=self.decoder_config).bert
            # self.lm_head = BertLMHeadModel.from_pretrained("bert-base-cased", config=self.decoder_config).cls
            print('self.transformer.config.is_decoder', self.transformer.config.is_decoder)

        self.learn_prior = True
        self.encoder = encoder_model_class[encoder_name][1].from_pretrained(encoder_name, config=self.encoder_config)
        self.encoder_prior = encoder_model_class[encoder_name][1](self.encoder_config)

        if 'deberta' in encoder_name:
            init_para_frompretrained_deberta(self.encoder_prior, self.encoder, share_para=True)
        else:
            init_para_frompretrained_roberta(self.encoder_prior, self.encoder, share_para=True)


        if ed_share:
            init_para_frompretrained_roberta(self.transformer, self.encoder, share_para=True, decoder=True)


    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(
            self,
            x_tokens=None,
            x_mask=None,
            y_tokens=None,
            y_mask=None,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        # x是context, y是ground truth, input 是teacher force
        hidden = self.encoder(input_ids=y_tokens, attention_mask=y_mask).last_hidden_state

        posterior, _ = self.averageSelfAttention(hidden, y_mask.squeeze(1).squeeze(1))
        posterior_mean = self.mean(posterior)
        posterior_logvar = self.logvar(posterior)

        # 用于计算KL divergence
        hidden2 = self.encoder_prior(input_ids=x_tokens, attention_mask=x_mask).last_hidden_state
        prior, _ = self.averageSelfAttention_prior(hidden2, x_mask.squeeze(1).squeeze(1))
        prior_mean = self.mean_prior(prior)
        prior_logvar = self.logvar_prior(prior)

        z = posterior_mean
        assert not torch.isnan(z).any(), 'training get nan z'

        transformer_outputs = self.transformer(input_ids,
                                               past_key_values=past_key_values,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               )

        hidden_states = transformer_outputs[0]  # [bz, sequence_length, 768]
        # add z
        input_proj = self.input_proj(z).unsqueeze(1)
        hidden_states = hidden_states + input_proj

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,)  # + transformer_outputs[1:]
        kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(
            0)
        outputs = outputs + (kl_loss, )

        return outputs

