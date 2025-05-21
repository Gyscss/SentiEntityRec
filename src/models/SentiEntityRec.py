import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv
import os
import pickle
# 以下代码会报错
# from models.base.layers import *
# from models.component.candidate_encoder import *
# from models.component.click_encoder import ClickEncoder
# from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
# from models.component.nce_loss import NCELoss
# from models.component.news_encoder import *
# from models.component.user_encoder import *

# 修改为
from .base.layers import *
from .component.candidate_encoder import *
from .component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from .component.entity_encoder import EntityEncoderWithSentiment, GlobalEntityEncoderWithSentiment
from .component.nce_loss import NCELoss
from .component.news_encoder import *
from .component.user_encoder import *

class SentiEntityRec(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity
        self.sentiment_dim = 3  # 添加情感向量维度配置

        self.news_dim =  cfg.model.head_num * cfg.model.head_dim
        self.entity_dim = cfg.model.entity_emb_dim

        # -------------------------- Model --------------------------
        # News Encoder
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)

        # GCN
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'),'x, index -> x'),
        ])
        # Entity
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

            self.local_entity_with_sentiment_encoder = Sequential('x, sentiment, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (EntityEncoderWithSentiment(cfg), 'x, sentiment, mask -> x'),   
            ])

            # self.local_entity_encoder = Sequential('x, mask', [
            #     (self.entity_embedding_layer, 'x -> x'),
            #     (EntityEncoder(cfg), 'x, mask -> x'),   
            # ])

            self.global_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),
            ])
        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)

        # User Encoder
        self.user_encoder = UserEncoder(cfg)
        
        # Candidate Encoder
        self.candidate_encoder = CandidateEncoder(cfg)

        # click prediction
        self.click_predictor = DotProduct()
        self.loss_fn = NCELoss()

    def _load_entity_sentiment(self, mode):
        """加载实体情感向量"""
    
        # 获取数据集名称
        print("查看self.cfg.dataset",self.cfg.dataset.dataset_name)
        dataset_name = self.cfg.dataset.dataset_name  # 从配置对象中获取数据集名称
        
        if mode == 'train':
            dataset_path = f"data/{dataset_name}/train/entity_sentiment.bin"
            print("查看是否是data/{dataset_name}/train/entity_sentiment.bin",dataset_path)
        elif mode == 'val':
            dataset_path = f"data/{dataset_name}/val/entity_sentiment.bin"
        elif mode == 'test':
            dataset_path = f"data/{dataset_name}/test/entity_sentiment.bin"
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # 加载实体情感向量字典
        entity_sentiment_dict = pickle.load(open(dataset_path, "rb"))
        
        # 打印字典的基本信息
        print("\n=== Entity Sentiment Dictionary Info ===")
        print(f"字典类型: {type(entity_sentiment_dict)}")
        print(f"字典大小: {len(entity_sentiment_dict)}")
        print(f"实体ID范围: [{min(entity_sentiment_dict.keys())}, {max(entity_sentiment_dict.keys())}]")
        
        # 打印前3个实体的情感向量作为示例
        print("\n前3个实体的情感向量示例:")
        for entity_id in sorted(list(entity_sentiment_dict.keys()))[:3]:
            print(f"实体ID {entity_id}: {entity_sentiment_dict[entity_id]}")
            print(f"情感向量类型: {type(entity_sentiment_dict[entity_id])}")
            print(f"情感向量长度: {len(entity_sentiment_dict[entity_id])}")
        
        # 将字典转换为张量
        max_entity_id = max(entity_sentiment_dict.keys())
        sentiment_dim = len(entity_sentiment_dict[1])  # 假设所有实体都有相同维度的情感向量
        print(f"\n情感向量维度: {sentiment_dim}")
        print(f"最大实体ID: {max_entity_id}")
        print(f"张量形状: [{max_entity_id + 1}, {sentiment_dim}]")
        print("说明：")
        print(f"- 索引0: 不使用（填充为0）")
        print(f"- 索引1-{max_entity_id}: 存储对应实体ID的情感向量")
        
        # 创建张量并确保数据类型正确
        entity_sentiment_tensor = torch.zeros(max_entity_id + 1, sentiment_dim, dtype=torch.float32)
        
        # 检查并转换每个情感向量
        for entity_id, sentiment in entity_sentiment_dict.items():
            if sentiment is None:
                print(f"警告：实体ID {entity_id} 的情感向量为None，使用默认值[0.0, 0.0, 0.0]")
                entity_sentiment_tensor[entity_id] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
                continue
                
            try:
                # 确保情感向量是浮点数列表，处理可能的None值
                sentiment_float = []
                for x in sentiment:
                    if x is None:
                        sentiment_float.append(0.0)
                    else:
                        sentiment_float.append(float(x))
                
                # 检查情感向量维度
                if len(sentiment_float) != sentiment_dim:
                    print(f"警告：实体ID {entity_id} 的情感向量维度不匹配")
                    print(f"期望维度: {sentiment_dim}, 实际维度: {len(sentiment_float)}")
                    print(f"情感向量内容: {sentiment_float}")
                    # 如果维度不足，用0填充
                    while len(sentiment_float) < sentiment_dim:
                        sentiment_float.append(0.0)
                    # 如果维度过多，截断
                    sentiment_float = sentiment_float[:sentiment_dim]
                        
                # 创建张量并赋值
                sentiment_tensor = torch.tensor(sentiment_float, dtype=torch.float32)
                # print(f"实体ID {entity_id} 的情感向量已成功转换: {sentiment_float}")
                # print(f"转换后的张量形状: {sentiment_tensor.shape}")
                entity_sentiment_tensor[entity_id] = sentiment_tensor
                
            except Exception as e:
                print(f"错误：处理实体ID {entity_id} 时出错: {str(e)}")
                print(f"情感向量内容: {sentiment}")
                print(f"情感向量类型: {type(sentiment)}")
                raise
            
        print(f"转换后的张量形状: {entity_sentiment_tensor.shape}")
        print("=== End Entity Sentiment Info ===\n")
            
        return entity_sentiment_tensor  # [num_entities, sentiment_dim]

    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, label=None):
        print("\n=== SentiEntityRec Forward Debug Info ===")
        print("1. 输入参数信息:")
        print(f"   subgraph.x shape: {subgraph.x.shape}")
        print(f"   mapping_idx shape: {mapping_idx.shape}")
        print(f"   candidate_news shape: {candidate_news.shape}")
        print(f"   candidate_entity shape: {candidate_entity.shape}")
        print(f"   entity_mask shape: {entity_mask.shape}")
        print(f"   label shape: {label.shape if label is not None else 'None'}")
        
        # 加载训练模式下的实体情感向量
        entity_sentiment = self._load_entity_sentiment('train')
        
        # -------------------------------------- clicked ----------------------------------
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0

        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        
        print("\n2. 处理点击新闻:")
        print(f"   batch_size: {batch_size}")
        print(f"   num_clicked: {num_clicked}")
        print(f"   token_dim: {token_dim}")
        
        # 分别提取实体和情感向量
        clicked_entity = subgraph.x[mapping_idx, -8:-3].long()  # 实体部分
        print(f"   clicked_entity shape: {clicked_entity.shape}")
        
        # 确保实体索引在有效范围内
        max_idx = entity_sentiment.shape[0] - 1
        clicked_entity = torch.clamp(clicked_entity, 0, max_idx)
        
        # 将entity_sentiment移动到与clicked_entity相同的设备上
        device = clicked_entity.device
        print(f"   device: {device}")
        entity_sentiment = entity_sentiment.to(device)
        
        # 使用实体ID获取对应的情感向量
        clicked_sentiment = entity_sentiment[clicked_entity]        # [batch_size, num_clicked, num_entities, sentiment_dim]
        print(f"   clicked_sentiment shape: {clicked_sentiment.shape}")

        # News Encoder + GCN
        x_flatten = subgraph.x.view(1, -1, token_dim)
        print(f"   x_flatten shape: {x_flatten.shape}")
        
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)
        print(f"   x_encoded shape: {x_encoded.shape}")    # [4106, 400]
        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)
        print(f"   graph_emb shape: {graph_emb.shape}")

        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)
        print(f"   clicked_origin_emb shape: {clicked_origin_emb.shape}")
        
        clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked, self.news_dim)
        print(f"   clicked_graph_emb shape: {clicked_graph_emb.shape}")

        # Attention pooling
        if self.use_entity:
            print("\n3. 处理实体信息:")
            clicked_entity_emb = self.local_entity_with_sentiment_encoder(clicked_entity, clicked_sentiment, None)
            print(f"   clicked_entity_emb shape: {clicked_entity_emb.shape}")
        else:
            clicked_entity_emb = None

        clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)
        print(f"   clicked_total_emb shape: {clicked_total_emb.shape}")
        
        user_emb = self.user_encoder(clicked_total_emb, mask)
        print(f"   user_emb shape: {user_emb.shape}")

        # ----------------------------------------- Candidate------------------------------------
        # "candidate_news shape"  # [8,5,38] 
        print("\n4. 处理候选新闻:")
        cand_title_emb = self.local_news_encoder(candidate_news)
        print(f"   cand_title_emb shape: {cand_title_emb.shape}")    # [8, 5, 400]
        
        if self.use_entity:
            origin_entity, neighbor_entity = candidate_entity.split([self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)
            print(f"   origin_entity shape: {origin_entity.shape}")
            print(f"   neighbor_entity shape: {neighbor_entity.shape}")
            
            origin_entity_sentiment = entity_sentiment[origin_entity]
            print(f"   origin_entity_sentiment shape: {origin_entity_sentiment.shape}")
            
            origin_entity = origin_entity.long()
            neighbor_entity = neighbor_entity.long()
            
            cand_origin_entity_emb = self.local_entity_with_sentiment_encoder(origin_entity, origin_entity_sentiment, None)
            print(f"   cand_origin_entity_emb shape: {cand_origin_entity_emb.shape}")    # [8, 5, 400]  
            
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)
            print(f"   cand_neighbor_entity_emb shape: {cand_neighbor_entity_emb.shape}")    # [8, 5, 400]  
        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb)
        print(f"   cand_final_emb shape: {cand_final_emb.shape}")
        
        # ----------------------------------------- Score ------------------------------------
        print("\n5. 计算分数和损失:")
        score = self.click_predictor(cand_final_emb, user_emb)
        print(f"   score shape: {score.shape}")
        
        loss = self.loss_fn(score, label)
        print(f"   loss shape: {loss.shape}")

        # # 确保所有参数都参与计算
        # if self.use_entity:
        #     # 添加一个小的正则化项，确保所有参数都参与计算
        #     reg_loss = 0.01 * (torch.norm(cand_origin_entity_emb) + torch.norm(cand_neighbor_entity_emb))
        #     loss = loss + reg_loss
        #     print(f"   reg_loss: {reg_loss.item()}")

        # # 确保所有参数都参与计算
        # if hasattr(self, 'entity_embedding_layer'):
        #     # 添加一个小的正则化项，确保entity_embedding_layer的参数参与计算
        #     reg_loss = 0.01 * torch.norm(self.entity_embedding_layer.weight)
        #     loss = loss + reg_loss
        #     print(f"   entity_embedding_reg_loss: {reg_loss.item()}")

        # print(f"   final loss: {loss.item()}")
        # print("=== End SentiEntityRec Forward Debug Info ===\n")

        return loss, score

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask):
        entity_sentiment = self._load_entity_sentiment('val')
        """验证过程的前向传播"""
        # 确保clicked_entity是long类型
        clicked_entity = clicked_entity.long()

        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        # 将entity_sentiment移动到与clicked_entity相同的设备上(后面的验证和测试也要改)
        device = clicked_entity.device
        print(f"entity_sentiment device: {entity_sentiment.device}")
        entity_sentiment = entity_sentiment.to(device)

        #--------------------Attention Pooling
        if self.use_entity:
            # 分别提取实体和情感向量

            # 使用实体ID获取对应的情感向量
            clicked_sentiment = entity_sentiment[clicked_entity]  # [batch_size, num_clicked, num_entities, sentiment_dim]
            
            entity_data = clicked_entity.unsqueeze(0)  # [1, num_news, num_entity]
            print(f"entity_data shape: {entity_data.shape}")
            
            clicked_entity_emb = self.local_entity_with_sentiment_encoder(entity_data, clicked_sentiment, None)
        else:
            clicked_entity_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)
        user_emb = self.user_encoder(clicked_final_emb)

        # ----------------------------------------- Candidate------------------------------------
        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            
            # 分割实体和邻居实体
            origin_entity, neighbor_entity = cand_entity_input.split([self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)
            
            # 不知道这里的clicked_entity是什么
            # 使用实体ID获取对应的情感向量
            origin_entity_sentiment = entity_sentiment[origin_entity]  # [batch_size, num_clicked, num_entities, sentiment_dim]
            
            # 打印维度信息
            print("\n=== Candidate Entity Encoder Debug Info ===")
            print(f"cand_entity_input shape: {cand_entity_input.shape}")
            print(f"origin_entity shape: {origin_entity.shape}")
            print(f"origin_entity_sentiment shape: {origin_entity_sentiment.shape}")
            
            # 确保维度正确
            if len(origin_entity_sentiment.shape) == 3:
                # 如果是三维的，添加batch维度
                origin_entity_sentiment = origin_entity_sentiment.unsqueeze(0)
                print(f"origin_entity_sentiment shape after unsqueeze: {origin_entity_sentiment.shape}")
            
            # 处理原始实体和邻居实体
            cand_origin_entity_emb = self.local_entity_with_sentiment_encoder(origin_entity, origin_entity_sentiment, None)
            print(f"cand_origin_entity_emb shape: {cand_origin_entity_emb.shape}")
            print("=== End Candidate Entity Encoder Debug Info ===\n")
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)
        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb, cand_neighbor_entity_emb)
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores

    def test_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask):
        entity_sentiment = self._load_entity_sentiment('test')
        # 确保clicked_entity是long类型
        clicked_entity = clicked_entity.long()
        
        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        # 将entity_sentiment移动到与clicked_entity相同的设备上(后面的验证和测试也要改)
        device = clicked_entity.device
        print(f"entity_sentiment device: {entity_sentiment.device}")
        entity_sentiment = entity_sentiment.to(device)

        #--------------------Attention Pooling
        if self.use_entity:
            # 分别提取实体和情感向量
            # 不知道这里的clicked_entity是什么
            # 使用实体ID获取对应的情感向量
            clicked_sentiment = entity_sentiment[clicked_entity]  # [batch_size, num_clicked, num_entities, sentiment_dim]
            
            entity_data = clicked_entity.unsqueeze(0)  # [1, num_news, num_entity]
            
            clicked_entity_emb = self.local_entity_with_sentiment_encoder(entity_data, clicked_sentiment, None)
        else:
            clicked_entity_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)
        user_emb = self.user_encoder(clicked_final_emb)

        # ----------------------------------------- Candidate------------------------------------
        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            
            # 分割实体和邻居实体
            origin_entity, neighbor_entity = cand_entity_input.split([self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)
            
            # 不知道这里的clicked_entity是什么
            # 使用实体ID获取对应的情感向量
            origin_entity_sentiment = entity_sentiment[origin_entity]  # [batch_size, num_clicked, num_entities, sentiment_dim]
            
            # 打印维度信息
            print("\n=== Candidate Entity Encoder Debug Info ===")
            print(f"cand_entity_input shape: {cand_entity_input.shape}")
            print(f"origin_entity shape: {origin_entity.shape}")
            print(f"origin_entity_sentiment shape: {origin_entity_sentiment.shape}")
            
            # 确保维度正确
            if len(origin_entity_sentiment.shape) == 3:
                # 如果是三维的，添加batch维度
                origin_entity_sentiment = origin_entity_sentiment.unsqueeze(0)
                print(f"origin_entity_sentiment shape after unsqueeze: {origin_entity_sentiment.shape}")
            
            # 处理原始实体和邻居实体
            cand_origin_entity_emb = self.local_entity_with_sentiment_encoder(origin_entity, origin_entity_sentiment, None)
            print(f"cand_origin_entity_emb shape: {cand_origin_entity_emb.shape}")
            print("=== End Candidate Entity Encoder Debug Info ===\n")
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)
        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb, cand_neighbor_entity_emb)
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores