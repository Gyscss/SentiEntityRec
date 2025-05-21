import torch
import torch.nn as nn
import numpy as np
# from models.base.layers import *    # # 这是源代码。但这里会报错
from ..base.layers import *
from torch_geometric.nn import Sequential


class CandidateEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_entity = cfg.model.use_entity

        self.entity_dim = 100
        self.news_dim = cfg.model.head_dim * cfg.model.head_num
        self.output_dim = cfg.model.head_dim * cfg.model.head_num

        if self.use_entity:
            self.atte = Sequential('a,b,c', [
                (lambda a,b,c: torch.stack([a,b,c], dim=-2).view(-1, 3, self.news_dim), 'a,b,c -> x'),
                AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                nn.Linear(self.news_dim, self.output_dim),
                nn.LeakyReLU(0.2),
            ])
        else:
            self.atte = Sequential('a,b,c', [
                (nn.Linear(self.news_dim, self.output_dim),'a -> x'),
                nn.LeakyReLU(0.2),
            ])


    def forward(self, candidate_emb, origin_emb=None, neighbor_emb=None):
        print("\n=== CandidateEncoder Forward Debug Info ===")
        print(f"1. Input shapes:")
        print(f"   candidate_emb: {candidate_emb.shape}")
        if origin_emb is not None:
            print(f"   origin_emb: {origin_emb.shape}")
        if neighbor_emb is not None:
            print(f"   neighbor_emb: {neighbor_emb.shape}")
            
        batch_size, num_news = candidate_emb.shape[0], candidate_emb.shape[1]
        
        if self.use_entity and origin_emb is not None and neighbor_emb is not None:
            # 调整维度以匹配
            # if len(origin_emb.shape) == 3:  # [batch_size, num_news, news_dim]
            #     origin_emb = origin_emb.view(-1, self.news_dim)  # [batch_size*num_news, news_dim]
            # if len(neighbor_emb.shape) == 3:  # [batch_size, num_news, news_dim]
            #     neighbor_emb = neighbor_emb.view(-1, self.news_dim)  # [batch_size*num_news, news_dim]
                
            print(f"2. After reshape:")
            print(f"   candidate_emb: {candidate_emb.shape}")
            print(f"   origin_emb: {origin_emb.shape}")
            print(f"   neighbor_emb: {neighbor_emb.shape}")
            
            result = self.atte(candidate_emb, origin_emb, neighbor_emb)
        else:
            result = self.atte(candidate_emb, None, None)
            
        print(f"3. After attention: {result.shape}")
        result = result.view(batch_size, num_news, self.output_dim)
        print(f"4. Final output: {result.shape}")
        print("=== End CandidateEncoder Forward Debug Info ===\n")
        
        return result