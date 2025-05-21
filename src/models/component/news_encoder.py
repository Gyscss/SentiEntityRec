import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path

import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *


class NewsEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        token_emb_dim = cfg.model.word_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        if cfg.dataset.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:
            self.word_encoder = nn.Embedding(glove_emb+1, 300, padding_idx=0)
            nn.init.uniform_(self.word_encoder.weight, -1.0, 1.0)

        self.view_size = [cfg.model.title_size, cfg.model.abstract_size]
        

        self.attention = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(token_emb_dim,
                                token_emb_dim,
                                token_emb_dim,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.news_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(self.news_dim,
                                cfg.model.attention_hidden_dim), 'x,mask -> x'),
            nn.LayerNorm(self.news_dim),
            # nn.Linear(self.news_dim, self.news_dim),
            # nn.LeakyReLU(0.2),
        ])


    def forward(self, news_input, mask=None):
        """
        处理新闻输入，返回新闻编码
        Args:
            news_input: shape [batch_size, num_news, title_size]
        Returns:
            news_emb: shape [batch_size, num_news, news_dim]
        """
        batch_size = news_input.shape[0]
        num_news = news_input.shape[1]

        # [batch_size * news_num, view_size, word_emb_dim]
        title_input, _, _, _, _ = news_input.split([self.view_size[0], 5, 1, 1, 1], dim=-1)
        
        # 确保输入是整数类型
        title_input = title_input.long()
        
        # 打印embedding层的大小和输入的范围
        print(f"Embedding weight shape: {self.word_encoder.weight.shape}")
        print(f"Title input min: {title_input.min().item()}, max: {title_input.max().item()}")
        print(f"Title input shape: {title_input.shape}")
        
        # 确保索引在有效范围内
        vocab_size = self.word_encoder.num_embeddings
        if title_input.max() >= vocab_size:
            print(f"Warning: Found index {title_input.max().item()} >= vocab_size {vocab_size}")
            title_input = torch.clamp(title_input, 0, vocab_size - 1)
        
        # 获取词嵌入
        title_word_emb = self.word_encoder(title_input.view(-1, self.view_size[0]))  # [batch_size * num_news, title_size, word_emb_dim]
        print("title_word_emb shape",title_word_emb.shape)  # [40, 30, 300]
        
        # 确保词嵌入是浮点类型
        title_word_emb = title_word_emb.float()
        
        # 应用注意力机制
        try:
            news_emb = self.attention(title_word_emb, mask)  # [batch_size, num_news, news_dim]
            print("news_emb shape",news_emb.shape)  # torch.Size([40, 400])
        except RuntimeError as e:
            print(f"Error in attention layer: {e}")
            print(f"title_word_emb shape: {title_word_emb.shape}")
            print(f"title_word_emb dtype: {title_word_emb.dtype}")
            print(f"title_word_emb device: {title_word_emb.device}")
            # 使用更安全的方式检查张量值
            try:
                min_val = title_word_emb.min().item()
                max_val = title_word_emb.max().item()
                print(f"title_word_emb min: {min_val}")
                print(f"title_word_emb max: {max_val}")
            except RuntimeError as e2:
                print(f"Error getting min/max values: {e2}")
            raise e
        
        news_emb = news_emb.view(batch_size, num_news, self.news_dim)   # torch.Size([8, 5, 400]
        print("news_emb.view(batch_size, num_news, self.news_dim) shape",news_emb.shape)
        return news_emb
