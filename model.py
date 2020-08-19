import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
	AdamW,
	get_linear_schedule_with_warmup
)

from src import (
	CONFIG_CLASSES,
	TOKENIZER_CLASSES,
	MODEL_FOR_SEQUENCE_CLASSIFICATION,
	MODEL_ORIGINER,
	init_logger,
	set_seed,
	compute_metrics
)
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from processor import seq_cls_processors as processors
from processor import seq_cls_output_modes as output_modes

class BASEELECTRA(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        outputs = self.dense(outputs[0][:,0,:])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)

        return result


class BASEELECTRA_COS(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(BASEELECTRA_COS, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embs = outputs[0]
        batch_size, seq_len, w2v_dim = embs.shape

        outputs = self.dense(embs[:,0,:])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss1 = loss_fct(outputs.view(-1, 2), labels.view(-1))


        x1 = embs[:,0,:].squeeze()
        x1 = x1.repeat(1, batch_size)
        x1 = x1.view(batch_size, batch_size, w2v_dim)
        x2 = embs[:,0,:].squeeze()
        x2 = x2.unsqueeze(0)
        x2 = x2.repeat(batch_size, 1, 1)
        label = torch.randint(0, 2, (batch_size,))
        # y = torch.randint(0,2,(batch_size, batch_size)) *2 -1
        y = labels.unsqueeze(0).repeat(batch_size, 1)
        for i, t in enumerate(y):
            y[i] = (t == t[i]).double() * 2 - 1
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=-1)
        loss2 = loss_fn(x1.view(-1, w2v_dim),
                       x2.view(-1, w2v_dim),
                       y.view(-1))

        result = (loss1+loss2, outputs)

        return result

class LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])

        outputs = self.dense(outputs[:,-1,:])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)

        return result

class LSTM_ATT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(outputs)
        # print(len(input_ids))
        # print(len(input_ids[0]))
        # print(len(outputs))
        # print(outputs[0].shape)
        outputs, (h, c) = self.lstm(outputs[0])
        # print("lstm")
        # print(len(outputs))
        # print(outputs.shape)

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)
        return result


class LSTM_ATT_v2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_v2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        # attention module
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dense_1 = nn.Linear(768, 100)
        self.dense_2 = nn.Linear(100, 1)

        # full connected
        self.fc = nn.Linear(768, 300)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(300, 2)

    def attention_net(self, lstm_outputs):
        M = self.tanh(self.dense_1(lstm_outputs))
        wM_output = self.dense_2(M).squeeze()
        a = self.softmax(wM_output)
        c = lstm_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])

        # attention
        attention_outputs = self.attention_net(outputs)

        fc_outputs = self.fc(attention_outputs)

        outputs = self.dropout(fc_outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result

class LSTM_ATT_DOT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT_DOT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class LSTM_ATT2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(LSTM_ATT2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w1 = nn.Parameter(torch.randn(1, 768, 10))
        self.att_w2 = nn.Parameter(torch.randn(1, 10, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w1.repeat(batch_size, 1, 1))
        att = torch.bmm(torch.tanh(att),
                        self.att_w2.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


#--KOSAC--

class KOSAC_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs = outputs[0] + polarity_emb_result/100 + intensity_emb_result/100
        outputs, _ = self.lstm(outputs)

        outputs = self.dense(outputs[:,-1,:])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KOSAC_LSTM_ATT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class KOSAC_LSTM_ATT_v2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT_v2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        # attention module
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dense_1 = nn.Linear(768, 100)
        self.dense_2 = nn.Linear(100, 1)

        # full connected
        self.fc = nn.Linear(768, 300)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(300, 2)

    def attention_net(self, lstm_outputs):
        M = self.tanh(self.dense_1(lstm_outputs))
        wM_output = self.dense_2(M).squeeze()
        a = self.softmax(wM_output)
        c = lstm_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        # attention
        attention_outputs = self.attention_net(outputs)

        fc_outputs = self.fc(attention_outputs)

        outputs = self.dropout(fc_outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result

class KOSAC_LSTM_ATT_DOT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT_DOT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(4, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100 + intensity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result

class KOSAC_LSTM_ATT_DOT_ML(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KOSAC_LSTM_ATT_DOT_ML, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100 + intensity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs, (h, c) = self.lstm(outputs[0][1:-1])
        attn_output, soft_attn_weights = self.attention_net(outputs, h)


        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        att_label = F.softmax((torch.abs(polarity_ids)+torch.abs(intensity_ids)).float(),dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        loss_att = nn.MSELoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1)) + loss_att(soft_attn_weights.squeeze(),att_label.long()).float()
        result = (loss, outputs)
        return result


#--KNU--

class KNU_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs = outputs[0] + polarity_emb_result/100 + intensity_emb_result/100
        outputs, _ = self.lstm(outputs)

        outputs = self.dense(outputs[:,-1,:])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result


class KNU_LSTM_ATT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len = input.shape

        final_h = final_h.squeeze()

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        # lstm_output(batch_size, seq_len, lstm_dir_dim)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        attn_output = self.re_attention(outputs, h, input_ids)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result


class KNU_LSTM_ATT_v2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT_v2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False, dropout=0.2)

        # attention module
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dense_1 = nn.Linear(768, 100)
        self.dense_2 = nn.Linear(100, 1)

        # full connected
        self.fc = nn.Linear(768, 300)

        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(300, 2)

    def attention_net(self, lstm_outputs):
        M = self.tanh(self.dense_1(lstm_outputs))
        wM_output = self.dense_2(M).squeeze()
        a = self.softmax(wM_output)
        c = lstm_outputs.transpose(1, 2).bmm(a.unsqueeze(-1)).squeeze()
        att_output = self.tanh(c)

        return att_output

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs = outputs[0] + polarity_emb_result / 100 + intensity_emb_result / 100
        outputs, (h, c) = self.lstm(outputs)

        # attention
        attention_outputs = self.attention_net(outputs)

        fc_outputs = self.fc(attention_outputs)

        outputs = self.dropout(fc_outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)

        return result

class KNU_LSTM_ATT_DOT(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT_DOT, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def forward(self, input_ids, attention_mask, labels, token_type_ids, polarity_ids):
        print(polarity_ids[0])
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output = self.attention_net(outputs, h)

        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

        result = (loss, outputs)
        return result

class KNU_LSTM_ATT_DOT_ML(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(KNU_LSTM_ATT_DOT_ML, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)

        # Embedding
        self.input_embedding = self.emb.embeddings.word_embeddings
        self.polarity_embedding = nn.Embedding(5, 768)
        self.intensity_embedding = nn.Embedding(5, 768)

        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)

        self.att_w = nn.Parameter(torch.randn(1, 768, 1))

    def attention_net(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input_ids, attention_mask, labels, token_type_ids,polarity_ids, intensity_ids):
        # embedding
        input_emb_result = self.input_embedding(input_ids)
        polarity_emb_result = self.polarity_embedding(polarity_ids)
        intensity_emb_result = self.intensity_embedding(intensity_ids)

        embedding_result = input_emb_result + polarity_emb_result / 100 + intensity_emb_result / 100

        outputs = self.emb(input_ids=None, attention_mask=attention_mask, token_type_ids=token_type_ids,inputs_embeds = embedding_result)
        outputs, (h, c) = self.lstm(outputs[0])
        attn_output, soft_attn_weights = self.attention_net(outputs, h)


        outputs = self.dense(attn_output)
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        att_label = F.softmax((torch.abs(polarity_ids)+torch.abs(intensity_ids)).float(),dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        loss_att = nn.MSELoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1)) + loss_att(soft_attn_weights.squeeze(),att_label.long()).float()
        result = (loss, outputs)
        return result

class EMB2_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(EMB2_LSTM, self).__init__()
        self.posemb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.negemb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        posoutputs = self.posemb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        posoutputs, (h, c) = self.lstm(posoutputs[0])

        posoutputs = self.dense(posoutputs[:,-1,:])
        posoutputs = self.dropout(posoutputs)
        posoutputs = self.out_proj(posoutputs)


        negoutputs = self.negemb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        negoutputs, (h, c) = self.lstm(negoutputs[0])

        negoutputs = self.dense(negoutputs[:,-1,:])
        negoutputs = self.dropout(negoutputs)
        negoutputs = self.out_proj(negoutputs)

        output = torch.cat((negoutputs, posoutputs), dim=1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, output)

        return result

class EMB1_LSTM2(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(EMB1_LSTM2, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.poslstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.posdense = nn.Linear(768, 768)
        self.posout_proj = nn.Linear(768, 1)

        self.dropout = nn.Dropout(0.2)

        self.neglstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.negdense = nn.Linear(768, 768)
        self.negout_proj = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, labels, token_type_ids):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        posoutputs, (h, c) = self.poslstm(outputs[0])

        posoutputs = self.posdense(posoutputs[:,-1,:])
        posoutputs = self.dropout(posoutputs)
        posoutputs = self.posout_proj(posoutputs)


        negoutputs, (h, c) = self.neglstm(outputs[0])

        negoutputs = self.negdense(negoutputs[:,-1,:])
        negoutputs = self.dropout(negoutputs)
        negoutputs = self.negout_proj(negoutputs)


        output = torch.cat((negoutputs, posoutputs), dim=1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, output)

        return result

class CHAR_LSTM(nn.Module):
    def __init__(self, model_type, model_name_or_path, config):
        super(CHAR_LSTM, self).__init__()
        self.emb = MODEL_ORIGINER[model_type].from_pretrained(
            model_name_or_path,
            config=config)
        self.char_emb_lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm = nn.LSTM(768, 768, batch_first=True, bidirectional=False)
        self.lstm_dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(768, 2)
        self.char_emb = None

    def forward(self, input_ids, attention_mask, labels, token_type_ids, char_token_data, word_token_data):
        # print(input_ids)
        outputs = self.emb(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        for line_index, line in enumerate(word_token_data):
            count = 0
            for word_tok in line:
                word_emb = self.char_emb_lstm(outputs[line_index,count:count+len(word_tok),:])

        outputs, (h, c) = self.lstm(outputs[0])

        outputs = self.dense(outputs[:,-1,:])
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
        # print(loss.shape)
        # print(loss)
        # print(len(outputs))
        # print(outputs.shape)

        result = (loss, outputs)

        return result

MODEL_LIST = {
    "BASEELECTRA": BASEELECTRA,
    "BASEELECTRA_COS": BASEELECTRA_COS,

    "LSTM": LSTM,
    "LSTM_ATT": LSTM_ATT,
    "LSTM_ATT_v2": LSTM_ATT_v2,
    "LSTM_ATT_DOT": LSTM_ATT_DOT,
    "LSTM_ATT2" : LSTM_ATT2,

    "LSTM_KOSAC": KOSAC_LSTM,
    "LSTM_ATT_KOSAC": KOSAC_LSTM_ATT,
    "LSTM_ATT_v2_KOSAC": KOSAC_LSTM_ATT_v2,
    "LSTM_ATT_DOT_KOSAC": KOSAC_LSTM_ATT_DOT,
    "KOSAC_LSTM_ATT_DOT_ML": KOSAC_LSTM_ATT_DOT_ML,

    "LSTM_KNU": KNU_LSTM,
    "LSTM_ATT_KNU": KNU_LSTM_ATT,
    "LSTM_ATT_v2_KNU": KNU_LSTM_ATT_v2,
    "LSTM_ATT_DOT_KNU": KNU_LSTM_ATT_DOT,
    "KOSAC_LSTM_ATT_DOT_ML": KNU_LSTM_ATT_DOT_ML,

    "CHAR_KOELECTRA": CHAR_LSTM,

    "EMB2_LSTM" : EMB2_LSTM,
    "EMB1_LSTM2" : EMB1_LSTM2
}
