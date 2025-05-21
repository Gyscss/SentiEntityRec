import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential

class ClickEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 设置新闻维度为400
        self.news_dim = 400
        # 是否使用实体信息
        self.use_entity = cfg.model.use_entity
        if self.use_entity:
            # 如果使用实体，则处理三个输入：新闻标题、图嵌入和实体嵌入
            self.atte = Sequential('a,b,c', [
                # 将三个输入堆叠在一起，并重塑维度
                (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
                # 使用注意力机制处理新闻标题
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
        else:
            # 如果不需要实体，则处理两个输入：新闻标题和图嵌入
            self.atte = Sequential('a,b', [
                # 将两个输入堆叠在一起，并重塑维度
                (lambda a,b: torch.stack([a,b], dim=-2).view(-1, 2, self.news_dim), 'a,b -> x'),
                # 使用注意力池化层处理
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
            ])
    
    def forward(self, clicke_title_emb, click_graph_emb, click_entity_emb=None):
        """
        前向传播函数
        Args:
            clicke_title_emb: 点击新闻的标题嵌入 [batch_size, num_news, news_dim]
            click_graph_emb: 点击新闻的图嵌入 [batch_size, num_news, news_dim]
            click_entity_emb: 点击新闻的实体嵌入 [batch_size, num_news, news_dim]，可选
        """
        # 获取批次大小和新闻数量
        batch_size, num_news = clicke_title_emb.shape[0], clicke_title_emb.shape[1]
        # 根据是否使用实体信息选择不同的处理方式
        if click_entity_emb is not None:
            # 如果有实体信息，传入三个输入
            result = self.atte(clicke_title_emb, click_graph_emb, click_entity_emb)
        else:
            # 如果没有实体信息，只传入两个输入
            result = self.atte(clicke_title_emb, click_graph_emb)

        # 返回处理后的结果，重塑维度为 [batch_size, num_news, news_dim]
        return result.view(batch_size, num_news, self.news_dim)
    
