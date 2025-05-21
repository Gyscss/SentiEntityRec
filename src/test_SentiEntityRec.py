import torch
import numpy as np
from models.SentiEntityRec import SentiEntityRec
from omegaconf import OmegaConf
import yaml

def test_validation_process():
    # 加载配置
    with open('configs/model/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    cfg = OmegaConf.create(config)
    
    # 初始化模型
    model = SentiEntityRec(cfg)
    model.eval()
    
    # 模拟输入数据
    batch_size = 4
    num_nodes = 10
    num_edges = 15
    num_entities = 100
    num_candidates = 5
    
    # 创建模拟数据
    subgraph = {
        'x': torch.randn(num_nodes, 768),  # 节点特征
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),  # 边索引
        'batch': torch.zeros(num_nodes, dtype=torch.long)  # 批次索引
    }
    
    mapping_idx = torch.randint(0, num_nodes, (batch_size, 2))  # 映射索引
    candidate_news = torch.randn(batch_size, num_candidates, 768)  # 候选新闻嵌入
    candidate_entity = torch.randint(0, num_entities, (batch_size, num_candidates))  # 候选实体
    entity_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)  # 实体掩码
    
    # 创建一个具体的clicked_entity示例
    clicked_entity = torch.tensor([5, 12, 8, 3], dtype=torch.long)  # 确保是long类型
    
    # 打印输入数据的形状和内容
    print("输入数据形状:")
    print(f"subgraph['x']: {subgraph['x'].shape}")
    print(f"subgraph['edge_index']: {subgraph['edge_index'].shape}")
    print(f"subgraph['batch']: {subgraph['batch'].shape}")
    print(f"mapping_idx: {mapping_idx.shape}")
    print(f"candidate_news: {candidate_news.shape}")
    print(f"candidate_entity: {candidate_entity.shape}")
    print(f"entity_mask: {entity_mask.shape}")
    print(f"clicked_entity: {clicked_entity.shape}")
    
    print("\nclicked_entity的具体内容:")
    print(f"数据: {clicked_entity}")
    print(f"数据类型: {clicked_entity.dtype}")
    print(f"设备: {clicked_entity.device}")
    
    try:
        # 调用validation_process
        scores = model.validation_process(
            subgraph, mapping_idx, clicked_entity,
            candidate_news, candidate_entity, entity_mask
        )
        
        print("\n输出数据形状:")
        print(f"scores: {scores.shape}")
        print("\n测试成功!")
        
    except Exception as e:
        print(f"\n测试失败! 错误信息: {str(e)}")
        raise e

if __name__ == "__main__":
    test_validation_process()

# 测试test_process方法
print("\nTesting test_process method:")
try:
    scores = model.test_process(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, clicked_entity)
    print("test_process output shape:", scores.shape)
    print("test_process output:", scores)
except Exception as e:
    print("Error in test_process:", e)