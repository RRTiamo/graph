import torch
from torch import nn
from transformers import AutoModel

from configuration import config


class SpellCheckModel(nn.Module):
    def __init__(self, free_param=False):
        super().__init__()
        # bert
        self.bert = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        # 冻结参数
        if free_param:
            for param in self.bert.parameters():
                param.requires_grad = False
        # linear
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.bert.config.pad_token_id)

    def forward(self, input_ids, attention_mask, label=None):
        output = self.bert(input_ids, attention_mask)
        # 取出最后一层的每一个时间步的隐藏状态
        last_hidden_state = output.last_hidden_state
        logits = self.linear(last_hidden_state)
        pred = torch.argmax(logits, dim=-1)
        # 掩码
        pred = pred.masked_fill(attention_mask == 0, self.bert.config.pad_token_id)
        # batch_size,seq_len
        res = {'pred': pred}
        if label is not None:
            # 计算损失 N*C N
            loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1))
            res['loss'] = loss
        return res
