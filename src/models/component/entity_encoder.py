import torch
import torch.nn as nn
from src.models.base.layers import *            # 使用相对于项目根目录的导入路径
from torch_geometric.nn import Sequential

class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = 400

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, int(self.entity_dim / cfg.model.head_dim), cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(self.entity_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Linear(self.entity_dim, self.news_dim),
            nn.LeakyReLU(0.2),
        ])


    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity, _ = entity_input.shape

        if entity_mask is not None:
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask.view(batch_size*num_news, num_entity)).view(batch_size, num_news, self.news_dim)
        else:
            result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), None).view(batch_size, num_news, self.news_dim)

        return result

class GlobalEntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, cfg.model.head_num, cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
        ])


    def forward(self, entity_input, entity_mask=None):
        print("\n=== GlobalEntityEncoder Forward Debug Info ===")
        print(f"1. Input shapes:")
        print(f"   entity_input: {entity_input.shape}")
        if entity_mask is not None:
            print(f"   entity_mask: {entity_mask.shape}")
        
        batch_size, num_news, num_entity, _ = entity_input.shape
        
        if entity_mask is not None:
            entity_mask = entity_mask.view(batch_size*num_news, num_entity)

        result = self.atte(entity_input.view(batch_size*num_news, num_entity, self.entity_dim), entity_mask).view(batch_size, num_news, self.news_dim)

        return result

# 以下是我加入的函数
class EntityEncoderWithSentiment(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.sentiment_dim = 3

        # 注意力层
        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(self.entity_dim + self.sentiment_dim, self.entity_dim + self.sentiment_dim, self.entity_dim + self.sentiment_dim, int((self.entity_dim + self.sentiment_dim) / cfg.model.head_num), cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Dropout(p=cfg.dropout_probability),
            (AttentionPooling(self.entity_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(self.entity_dim),
            nn.Linear(self.entity_dim, self.news_dim),
            nn.LeakyReLU(0.2),
        ])

        # # 对于多头注意力层的讲解，可以用于写论文
        # 假设：
        # batch_size = 8
        # num_news = 50
        # num_entity = 5
        # entity_dim = 100
        # sentiment_dim = 3
        # news_dim = 400
        # head_dim = 50
        # head_num = 8

        # # 1. 输入
        # entity_features: [400, 5, 103]  # [batch_size*num_news, num_entity, entity_dim+sentiment_dim]
        # entity_mask: [400, 5]          # [batch_size*num_news, num_entity]

        # 2. 第一个Dropout层
        # 维度保持不变：[400, 5, 103]

        # 3. 多头注意力层
        # 内部过程：
        # a. 将输入分成多个头
        # [400, 5, 103] -> [400, 5, 8, 13]  # 8个头，每个头13维
        # b. 计算注意力分数
        # [400, 5, 5]  # 每个实体与其他实体的注意力分数
        # c. 应用mask
        # [400, 5, 5]  # 被mask的位置设为0
        # d. 加权求和
        # [400, 5, 103]  # 输出维度与输入相同

        # 4. LayerNorm层
        # 维度保持不变：[400, 5, 103]

        # 5. 第二个Dropout层
        # 维度保持不变：[400, 5, 103]

        # 6. 注意力池化层
        # 将多个实体池化为一个向量
        # [400, 5, 103] -> [400, 103]

        # 7. LayerNorm层
        # 维度保持不变：[400, 103]

        # 8. 线性投影层
        # 将103维投影到400维
        # [400, 103] -> [400, 400]

        # 9. LeakyReLU激活
        # 维度保持不变：[400, 400]

        # 10. 在forward方法中reshape
        # result.view(batch_size, num_news, self.news_dim)  # [8, 50, 400]

    def forward(self, entity_input, entity_sentiment, entity_mask=None):
        """
        处理实体输入和情感向量
        Args:
            entity_input: 实体特征 [batch_size, num_news, num_entity, entity_dim]
            entity_sentiment: 每个实体的情感向量 [batch_size, num_news, num_entity, sentiment_dim]
            entity_mask: 实体mask [batch_size, num_news, num_entity]
        """
        batch_size, num_news, num_entity, _ = entity_input.shape
        print("\n=== EntityEncoderWithSentiment Forward Debug Info ===")
        print(f"1. Input shapes:")
        print(f"   entity_input: {entity_input.shape}")
        print(f"   entity_sentiment: {entity_sentiment.shape}")
        
        # 检查输入维度
        assert entity_sentiment.shape[:-1] == entity_input.shape[:-1], \
            f"实体特征和情感向量维度不匹配: {entity_input.shape} vs {entity_sentiment.shape}"
        
        # 1. 直接拼接实体特征和情感向量
        entity_features = torch.cat([entity_input, entity_sentiment], dim=-1)  # [batch_size, num_news, num_entity, entity_dim+sentiment_dim]
        print(f"2. After concatenation: {entity_features.shape}")
        
        # 2. 重塑维度以进行注意力计算
        entity_features = entity_features.view(batch_size * num_news, num_entity, self.entity_dim + self.sentiment_dim)
        print(f"3. After reshape: {entity_features.shape}")
        
        # 3. 应用注意力机制
        if entity_mask is not None:
            # 检查mask维度
            assert entity_mask.shape == (batch_size, num_news, num_entity), \
                f"mask维度不匹配: {entity_mask.shape} vs ({batch_size}, {num_news}, {num_entity})"
            entity_mask = entity_mask.view(batch_size * num_news, num_entity)
            print(f"4. Using mask: {entity_mask.shape}")
            result = self.atte(entity_features, entity_mask)
        else:
            print("4. No mask used")
            result = self.atte(entity_features, None)
        
        print(f"5. After attention: {result.shape}")
        
        # 4. 重塑回原始维度
        result = result.view(batch_size, num_news, self.news_dim)
        print(f"6. Final output: {result.shape}")
        print("=== End EntityEncoderWithSentiment Forward Debug Info ===\n")
        
        return result


# 这里是我想单独计算每个实体与其情感向量
# class EntityEncoderWithSentiment(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.entity_dim = cfg.model.entity_emb_dim
#         self.news_dim = cfg.model.head_num * cfg.model.head_dim
#         self.sentiment_dim = 3
#
#         # 情感投影层
#         self.sentiment_proj = nn.Linear(self.sentiment_dim, self.entity_dim)
#
#         # 注意力层
#         self.atte = Sequential('x, mask', [
#             (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
#             (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, cfg.model.head_num,
#                                 cfg.model.head_dim), 'x,x,x,mask -> x'),
#             nn.LayerNorm(self.news_dim),
#             nn.Dropout(p=cfg.dropout_probability),
#             (AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
#             nn.LayerNorm(self.news_dim),
#         ])
#
#     def forward(self, entity_input, sentiment_vector, entity_mask=None):
#         batch_size, num_news, num_entity, _ = entity_input.shape
#
#         # 1. 处理情感向量
#         sentiment_proj = self.sentiment_proj(sentiment_vector.float())  # [batch_size, num_news, num_entity, entity_dim]
#
#         # 2. 将情感向量与实体特征结合
#         entity_features = entity_input + sentiment_proj  # 简单的加法融合
#
#         # 3. 重塑维度以进行注意力计算
#         entity_features = entity_features.view(batch_size * num_news, num_entity, self.entity_dim)
#         if entity_mask is not None:
#             entity_mask = entity_mask.view(batch_size * num_news, num_entity)
#
#         # 4. 应用注意力机制
#         result = self.atte(entity_features, entity_mask)
#
#         # 5. 重塑回原始维度
#         result = result.view(batch_size, num_news, self.news_dim)
#
#         return result


class GlobalEntityEncoderWithSentiment(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.sentiment_dim = 3

        # 情感投影层
        self.sentiment_proj = nn.Linear(self.sentiment_dim, self.entity_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.news_dim + self.entity_dim, self.news_dim),  # 实体和情感分别投影
            nn.LayerNorm(self.news_dim),
            nn.LeakyReLU(0.2)
        )

        self.atte = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(self.entity_dim, self.entity_dim, self.entity_dim, cfg.model.head_num, cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
            nn.Dropout(p=cfg.dropout_probability),
            (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x, mask-> x'),
            nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
        ])

    def forward(self, entity_input, sentiment_vector, entity_mask=None):
        print("GlobalEntityEncoderWithSentiment forward:")
        print(f"entity_input shape: {entity_input.shape}, dtype: {entity_input.dtype}")
        print(f"sentiment_vector shape: {sentiment_vector.shape}, dtype: {sentiment_vector.dtype}")
        if entity_mask is not None:
            print(f"entity_mask shape: {entity_mask.shape}, dtype: {entity_mask.dtype}")
            
        batch_size, num_news, num_entity, _ = entity_input.shape
        
        # 1. 实体注意力处理
        if entity_mask is not None:
            # 调整 entity_mask 的维度以匹配 entity_input
            if entity_mask.size(-1) != num_entity:
                # 如果 entity_mask 的维度小于 num_entity，用 False 填充
                if entity_mask.size(-1) < num_entity:
                    padding = torch.zeros(batch_size, num_news, num_entity - entity_mask.size(-1), 
                                        dtype=entity_mask.dtype, device=entity_mask.device)
                    entity_mask = torch.cat([entity_mask, padding], dim=-1)
                # 如果 entity_mask 的维度大于 num_entity，截断
                else:
                    entity_mask = entity_mask[:, :, :num_entity]
            
            # 重塑输入
            entity_input = entity_input.view(batch_size * num_news, num_entity, self.entity_dim)
            entity_mask = entity_mask.view(batch_size * num_news, num_entity)
            print(f"Before attention - entity_input shape: {entity_input.shape}, dtype: {entity_input.dtype}")
            print(f"Before attention - entity_mask shape: {entity_mask.shape}, dtype: {entity_mask.dtype}")
            result = self.atte(entity_input, entity_mask)
        else:
            # 重塑输入
            entity_input = entity_input.view(batch_size * num_news, num_entity, self.entity_dim)
            print(f"Before attention - entity_input shape: {entity_input.shape}, dtype: {entity_input.dtype}")
            result = self.atte(entity_input, None)
            
        result = result.view(batch_size, num_news, self.news_dim)
        print(f"After attention - result shape: {result.shape}, dtype: {result.dtype}")
        
        # 2. 情感向量处理
        sentiment_proj = self.sentiment_proj(sentiment_vector.float())  # [batch_size, num_news, sentiment_dim] -> [batch_size, num_news, entity_dim]
        print(f"After sentiment projection - sentiment_proj shape: {sentiment_proj.shape}, dtype: {sentiment_proj.dtype}")
        
        # 3. 拼接实体和情感特征
        combined = torch.cat([result, sentiment_proj], dim=-1)  # [batch_size, num_news, news_dim + entity_dim]
        print(f"After concatenation - combined shape: {combined.shape}, dtype: {combined.dtype}")
        
        # 4. 输出处理
        result = self.output_layer(combined)  # [batch_size, num_news, news_dim]
        print(f"Final output shape: {result.shape}, dtype: {result.dtype}")
        
        return result