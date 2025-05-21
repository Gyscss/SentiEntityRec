import torch
import torch.nn.functional as f
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from abc import abstractmethod
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np
import pickle
from pathlib import Path


class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.local_rank = local_rank
        self.cfg = cfg
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    @staticmethod
    def pad_to_fix_len(x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):
        line = line.strip().split('\t')
        
        # 处理点击历史
        click_id = line[3].split()[-self.cfg.model.his_size:]
        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index]

        # 处理候选新闻（正样本和负样本）
        pos_news_id = line[4].split()  # 正样本新闻ID
        neg_news_ids = line[5].split()  # 负样本新闻ID列表
        candidate_index = self.trans_to_nindex(pos_news_id + neg_news_ids)
        candidate_input = self.news_input[candidate_index]

        labels = 0

        # 处理标签（正样本为1，负样本为0）
        # labels = np.array([1] + [0] * len(neg_news_ids))

        # # 将numpy数组转换为PyTorch张量后再检查
        # candidate_input = torch.from_numpy(candidate_input)
        # if torch.isnan(candidate_input).any() or torch.isinf(candidate_input).any():
        #     print(f"Warning: Invalid values in candidate_input")
        #     candidate_input = torch.nan_to_num(candidate_input, nan=0.0, posinf=0.0, neginf=0.0)
        #     candidate_input = candidate_input.numpy()

        return clicked_input, clicked_mask, candidate_input, clicked_index, candidate_index, labels

    @abstractmethod
    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)
    
    
class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input,
                 local_rank, cfg, neighbor_dict=None, news_graph=None, entity_neighbors=None):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True) if news_graph is not None else None
        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.sum_num_news = 0
        
        # 先不在实体图里用情感向量
        # 加载情感向量
        # sentiment_path = Path(cfg.dataset.train_dir) / "entity_sentiment.bin"
        # if sentiment_path.exists():
        #     raw_sentiment = pickle.load(open(sentiment_path, "rb"))
        #     # 处理情感向量，确保数据有效性
        #     self.entity_sentiment = {}
        #     for entity_id, sentiment in raw_sentiment.items():
        #         if sentiment is not None and isinstance(sentiment, (list, tuple)) and len(sentiment) == 3:
        #             try:
        #                 # 确保所有值都是有效的浮点数
        #                 valid_sentiment = [float(x) for x in sentiment]
        #                 self.entity_sentiment[entity_id] = valid_sentiment
        #             except (ValueError, TypeError):
        #                 continue
        #     print(f"[Train] Loaded {len(self.entity_sentiment)} valid entity sentiment vectors from {sentiment_path}")
        # else:
        #     self.entity_sentiment = {}
        #     print(f"[Train] Warning: No entity sentiment vectors found at {sentiment_path}")
        
        # # 默认情感向量
        # self.default_sentiment = [0.0, 0.0, 0.0]

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        
        # 修改这里：处理候选新闻（正样本和负样本）
        pos_news_id = line[4].split()  # 正样本新闻ID
        neg_news_id = line[5].split()  # 负样本新闻ID列表
        

        # 我不知道下面注释掉的代码是干嘛的
        # # 检查news_index中的映射
        # candidate_index = self.trans_to_nindex(candidate_news_ids)
        # print(f"Original news IDs: {candidate_news_ids}")
        # print(f"Mapped indices: {candidate_index}")
        
        # # 检查映射后的索引是否有效
        # if any(idx < 0 for idx in candidate_index):
        #     print(f"Warning: Found negative indices in candidate_index: {candidate_index}")
        #     candidate_index = [max(0, idx) for idx in candidate_index]
        
        # candidate_input = self.news_input[candidate_index]
        
        # # 将numpy数组转换为PyTorch张量后再检查
        # candidate_tensor = torch.from_numpy(candidate_input)
        # if torch.isnan(candidate_tensor).any() or torch.isinf(candidate_tensor).any():
        #     print(f"Warning: Invalid values in candidate_input")
        #     candidate_tensor = torch.nan_to_num(candidate_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        #     candidate_input = candidate_tensor.numpy()
        
        # # 确保所有值都是非负的
        # if (candidate_tensor < 0).any():
        #     print(f"Warning: Found negative values in candidate_input")
        #     print(f"Negative values at positions: {torch.where(candidate_tensor < 0)}")
        #     candidate_tensor = torch.clamp(candidate_tensor, min=0)
        #     candidate_input = candidate_tensor.numpy()
        
        # 修改这里：生成标签（正样本为1，负样本为0）
        # labels = np.array([1] + [0] * len(neg_news_id))


        # ------------------ Clicked News ----------------------
        # ------------------ News Subgraph ---------------------
        top_k = len(click_id)
        click_idx = self.trans_to_nindex(click_id)
        source_idx = click_idx

        # 构建子图
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        
        print("\n=== Mapping Info of TrainGraphDataset ===")
        # print(f"Original click_id: {click_id}")  # 原始的新闻ID
        # print(f"click_idx after neighbor sampling: {click_idx}")  # 采样后的新闻索引
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, self.sum_num_news)
        print(f"mapping_idx shape: {mapping_idx.shape}")  # mapping_idx的形状
        # print(f"mapping_idx content: {mapping_idx}")  # mapping_idx的内容
        print("=== End Mapping Info of TrainGraphDataset ===\n")
        padded_mapping_idx = f.pad(mapping_idx, (self.cfg.model.his_size-len(mapping_idx), 0), "constant", -1)

        # ------------------ Candidate News ----------------------
        label = 0
        sample_news = self.trans_to_nindex(pos_news_id + neg_news_id)
        candidate_input = self.news_input[sample_news]

    # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]  #[5, 5]
            candidate_neighbor_entity = np.zeros(((self.cfg.npratio+1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64) # [5*5, 20]
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio+1, self.cfg.model.entity_size *self.cfg.model.entity_neighbors) # [5, 5*20]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)


        # 暂时不考虑带情感向量的实体图
        # ------------------ Entity and Sentiment --------------------
        # if self.cfg.model.use_entity:
        #     # 处理原始实体
        #     origin_entity = candidate_input[:, -8:-3]  # 实体部分在倒数第8到第3个位置
            
        #     # 获取实体的情感向量并组合
        #     entity_with_sentiment = []
        #     for entities in origin_entity:
        #         entity_group = []
        #         for entity_id in entities:
        #             if entity_id in self.entity_sentiment:
        #                 # 将实体ID和情感向量分开存储
        #                 entity_group.append([int(entity_id), *[float(x) for x in self.entity_sentiment[entity_id]]])
        #             else:
        #                 entity_group.append([int(entity_id), *[float(x) for x in self.default_sentiment]])
        #         entity_with_sentiment.append(entity_group)
        #     entity_with_sentiment = np.array(entity_with_sentiment)
            
        #     # 处理邻居实体
        #     candidate_neighbor_entity = np.zeros(
        #         (len(candidate_index) * self.cfg.model.entity_size, 
        #          self.cfg.model.entity_neighbors), 
        #         dtype=np.int64
        #     )
        #     for cnt, idx in enumerate(origin_entity.flatten()):
        #         if idx == 0:
        #             continue
        #         entity_dict_length = len(self.entity_neighbors[idx])
        #         if entity_dict_length == 0:
        #             continue
        #         valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
        #         candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

        #     candidate_neighbor_entity = candidate_neighbor_entity.reshape(
        #         len(candidate_index),
        #         self.cfg.model.entity_size * self.cfg.model.entity_neighbors
        #     )
            
        #     # 处理邻居实体的情感向量并组合
        #     neighbor_with_sentiment = []
        #     for i, neighbors in enumerate(candidate_neighbor_entity):
        #         neighbor_group = []
        #         for neighbor_id in neighbors:
        #             if neighbor_id in self.entity_sentiment:
        #                 # 将实体ID和情感向量分开存储
        #                 neighbor_group.append([int(neighbor_id), *[float(x) for x in self.entity_sentiment[neighbor_id]]])
        #             else:
        #                 neighbor_group.append([int(neighbor_id), *[float(x) for x in self.default_sentiment]])
        #         neighbor_with_sentiment.append(neighbor_group)
        #     neighbor_with_sentiment = np.array(neighbor_with_sentiment)
            
        #     entity_mask = candidate_neighbor_entity.copy()
        #     entity_mask[entity_mask > 0] = 1
            
        #     # 合并所有信息
        #     # 将entity_with_sentiment和neighbor_with_sentiment展平后再拼接
        #     entity_flat = entity_with_sentiment.reshape(len(candidate_index), -1)
        #     neighbor_flat = neighbor_with_sentiment.reshape(len(candidate_index), -1)
        #     candidate_entity = np.concatenate((entity_flat, neighbor_flat), axis=-1)
        # else:
        #     candidate_entity = np.zeros(1, dtype=np.int64)
        #     entity_mask = np.zeros(1)

        # self.sum_num_news += sub_news_graph.num_nodes
        
        return sub_news_graph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news + sub_news_graph.num_nodes
        
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset: 
            subset = [0]
            
        subset = torch.tensor(subset, dtype=torch.long, device=device)
        
        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(
            unique_subset, 
            self.news_graph.edge_index, 
            self.news_graph.edge_attr, 
            relabel_nodes=True, 
            num_nodes=self.news_graph.num_nodes
        )
        
        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k]+sum_num_nodes
    
    @abstractmethod
    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as file:
                for line in file:
                    # if line.strip().split('\t')[3]:
                    result = self.line_mapper(line, sum_num_news)
                    (sub_newsgraph, padded_mapping_idx, candidate_input, 
                     candidate_entity, entity_mask, label, sum_num_news) = result

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        # 优化labels的创建方式
                        labels = torch.from_numpy(np.array(labels)).long()
                        yield (batch, mappings, candidates, candidate_entity_list, 
                               entity_mask_list, labels)
                        (clicked_graphs, mappings, candidates, labels, 
                         candidate_entity_list, entity_mask_list) = [], [], [], [], [], []
                        sum_num_news = 0

                if len(clicked_graphs) > 0:
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    # 优化labels的创建方式
                    labels = torch.from_numpy(np.array(labels)).long()

                    yield (batch, mappings, candidates, candidate_entity_list, 
                           entity_mask_list, labels)
                    file.seek(0)


class ValidDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.local_rank = local_rank
        self.cfg = cfg
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    @staticmethod
    def pad_to_fix_len(x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):
        line = line.strip().split('\t')
        
        # 处理点击历史
        click_id = line[3].split()[-self.cfg.model.his_size:]
        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index]

        # 处理候选新闻（正样本和负样本）
        pos_news_id = line[4].split()  # 正样本新闻ID
        neg_news_ids = line[5].split()  # 负样本新闻ID列表
        candidate_index = self.trans_to_nindex(pos_news_id + neg_news_ids)
        candidate_input = self.news_input[candidate_index]

        labels = 0

        # 处理标签（正样本为1，负样本为0）
        # labels = np.array([1] + [0] * len(neg_news_ids))

        # # 将numpy数组转换为PyTorch张量后再检查
        # candidate_input = torch.from_numpy(candidate_input)
        # if torch.isnan(candidate_input).any() or torch.isinf(candidate_input).any():
        #     print(f"Warning: Invalid values in candidate_input")
        #     candidate_input = torch.nan_to_num(candidate_input, nan=0.0, posinf=0.0, neginf=0.0)
        #     candidate_input = candidate_input.numpy()

        return clicked_input, clicked_mask, candidate_input, clicked_index, candidate_index, labels

    @abstractmethod
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:  # 确保有点击历史
                yield self.line_mapper(line)

    # @abstractmethod
    # def __iter__(self):
    #     file_iter = open(self.filename)
    #     return map(self.line_mapper, file_iter)

class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input,
                 local_rank, cfg, neighbor_dict, news_graph, news_entity, entity_neighbors):
        super().__init__(
            filename, news_index, news_input,
            local_rank, cfg, neighbor_dict, news_graph, entity_neighbors
        )
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity

        # 先不考虑情感向量
        # # 加载验证集的情感向量
        # sentiment_path = Path(cfg.dataset.val_dir) / "entity_sentiment.bin"
        # if sentiment_path.exists():
        #     self.entity_sentiment = pickle.load(open(sentiment_path, "rb"))
        #     print(f"[Val] Loaded entity sentiment vectors from {sentiment_path}")
        # else:
        #     self.entity_sentiment = {}
        #     print(f"[Val] Warning: No entity sentiment vectors found at {sentiment_path}")

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        click_idx = self.trans_to_nindex(click_id)
        clicked_entity = self.news_input[click_idx]
        
        # 获取点击新闻实体的情感向量
        # sentiment_vectors = []
        # for entities in clicked_entity:
        #     entity_sentiments = []
        #     for entity_id in entities:
        #         if entity_id in self.entity_sentiment:
        #             entity_sentiments.append(self.entity_sentiment[entity_id])
        #         else:
        #             entity_sentiments.append(self.default_sentiment)
        #     sentiment_vectors.append(entity_sentiments)
        # clicked_sentiment = np.array(sentiment_vectors)
        
        # 构建子图
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        
        print("\n=== Mapping Info of ValidGraphDataset ===")
        # print(f"Original click_id: {click_id}")  # 原始的新闻ID
        # print(f"click_idx after neighbor sampling: {click_idx}")  # 采样后的新闻索引
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)
        print(f"mapping_idx shape: {mapping_idx.shape}")  # mapping_idx的形状
        # print(f"mapping_idx content: {mapping_idx}")  # mapping_idx的内容
        print("=== End Mapping Info of ValidGraphDataset ===\n")

        # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]
        
        #这里有异议，到底是上面的还是下面的
        # 修改这里：处理候选新闻（正样本和负样本）
        # pos_news_id = line[4].split()  # 正样本新闻ID
        # neg_news_ids = line[5].split()  # 负样本新闻ID列表
        # candidate_news_ids = [pos_news_id] + neg_news_ids
        # candidate_index = self.trans_to_nindex(candidate_news_ids)
        # candidate_input = self.news_input[candidate_index]
        
        # # 检查candidate_input的值范围
        # if torch.isnan(candidate_input).any() or torch.isinf(candidate_input).any():
        #     print(f"Warning: Invalid values in candidate_input")
        #     candidate_input = torch.nan_to_num(candidate_input, nan=0.0, posinf=0.0, neginf=0.0)
        
        # # 确保所有值都是非负的
        # candidate_input = torch.clamp(candidate_input, min=0)
        
        # 修改这里：生成标签（正样本为1，负样本为0）
        # labels = np.array([1] + [0] * len(neg_news_ids))

        if self.cfg.model.use_entity:
            # 处理原始实体
            origin_entity = self.news_input[candidate_index]
            
            # 先不考虑情感向量
            # 获取候选新闻实体的情感向量
            # candidate_sentiment = []
            # for entities in origin_entity:
            #     entity_sentiments = []
            #     for entity_id in entities:
            #         if entity_id in self.entity_sentiment:
            #             entity_sentiments.append(self.entity_sentiment[entity_id])
            #         else:
            #             entity_sentiments.append(self.default_sentiment)
            #     candidate_sentiment.append(entity_sentiments)
            # candidate_sentiment = np.array(candidate_sentiment)
            
            # 处理邻居实体
            candidate_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, 
                 self.cfg.model.entity_neighbors), 
                dtype=np.int64
            )
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]
            
            candidate_neighbor_entity = candidate_neighbor_entity.reshape(
                len(candidate_index),
                self.cfg.model.entity_size * self.cfg.model.entity_neighbors
            )
            
            # 处理邻居实体的情感向量
            # neighbor_sentiment = np.zeros((len(candidate_index),
            #                                self.cfg.model.entity_size * self.cfg.model.entity_neighbors, 3))
            # for i, neighbors in enumerate(candidate_neighbor_entity):
            #     for j, neighbor_id in enumerate(neighbors):
            #         if neighbor_id in self.entity_sentiment:
            #             neighbor_sentiment[i, j] = self.entity_sentiment[neighbor_id]
            #         else:
            #             neighbor_sentiment[i, j] = self.default_sentiment
            
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            
            # 合并所有信息
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
            # candidate_sentiment = np.concatenate((candidate_sentiment, neighbor_sentiment), axis=-1)
            # candidate_entity = np.concatenate((candidate_entity, candidate_sentiment), axis=-1)
            
            # 合并点击新闻的实体和情感向量
            # clicked_entity = np.concatenate((clicked_entity, clicked_sentiment), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)
            # clicked_entity = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])
        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels
    
    @abstractmethod
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:  # 确保有点击历史
                result = self.line_mapper(line)
                (batch, mapping_idx, clicked_entity, candidate_input,
                 candidate_entity, entity_mask, labels) = result
                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels
            else:
                continue  # 如果没有点击历史，跳过这一行


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class TestDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.local_rank = local_rank
        self.cfg = cfg
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    @staticmethod
    def pad_to_fix_len(x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):
        line = line.strip().split('\t')
        
        # 处理点击历史
        click_id = line[3].split()[-self.cfg.model.his_size:]
        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index]

        # 处理候选新闻（正样本和负样本）
        pos_news_id = line[4].split()  # 正样本新闻ID
        neg_news_ids = line[5].split()  # 负样本新闻ID列表
        candidate_index = self.trans_to_nindex(pos_news_id + neg_news_ids)
        candidate_input = self.news_input[candidate_index]

        labels = 0

        # 处理标签（正样本为1，负样本为0）
        # labels = np.array([1] + [0] * len(neg_news_ids))

        # # 将numpy数组转换为PyTorch张量后再检查
        # candidate_input = torch.from_numpy(candidate_input)
        # if torch.isnan(candidate_input).any() or torch.isinf(candidate_input).any():
        #     print(f"Warning: Invalid values in candidate_input")
        #     candidate_input = torch.nan_to_num(candidate_input, nan=0.0, posinf=0.0, neginf=0.0)
        #     candidate_input = candidate_input.numpy()

        return clicked_input, clicked_mask, candidate_input, clicked_index, candidate_index, labels

    @abstractmethod
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:  # 确保有点击历史
                yield self.line_mapper(line)


class TestGraphDataset(ValidGraphDataset):
    """测试图数据集，继承自ValidGraphDataset"""
    def __init__(self, filename, news_index, news_input,
                 local_rank, cfg, neighbor_dict, news_graph, news_entity, entity_neighbors):
        super().__init__(
            filename, news_index, news_input,
            local_rank, cfg, neighbor_dict, news_graph, entity_neighbors
        )
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity

        # 先不考虑情感向量
        # # 加载验证集的情感向量
        # sentiment_path = Path(cfg.dataset.val_dir) / "entity_sentiment.bin"
        # if sentiment_path.exists():
        #     self.entity_sentiment = pickle.load(open(sentiment_path, "rb"))
        #     print(f"[Val] Loaded entity sentiment vectors from {sentiment_path}")
        # else:
        #     self.entity_sentiment = {}
        #     print(f"[Val] Warning: No entity sentiment vectors found at {sentiment_path}")

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        click_idx = self.trans_to_nindex(click_id)
        clicked_entity = self.news_input[click_idx]
        
        # 获取点击新闻实体的情感向量
        # sentiment_vectors = []
        # for entities in clicked_entity:
        #     entity_sentiments = []
        #     for entity_id in entities:
        #         if entity_id in self.entity_sentiment:
        #             entity_sentiments.append(self.entity_sentiment[entity_id])
        #         else:
        #             entity_sentiments.append(self.default_sentiment)
        #     sentiment_vectors.append(entity_sentiments)
        # clicked_sentiment = np.array(sentiment_vectors)
        
        # 构建子图
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        
        print("\n=== Mapping Info of TestGraphDataset ===")
        # print(f"Original click_id: {click_id}")  # 原始的新闻ID
        # print(f"click_idx after neighbor sampling: {click_idx}")  # 采样后的新闻索引
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)
        print(f"mapping_idx shape: {mapping_idx.shape}")  # mapping_idx的形状
        # print(f"mapping_idx content: {mapping_idx}")  # mapping_idx的内容
        print("=== End Mapping Info of TestGraphDataset ===\n")

        # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]
        
        #这里有异议，到底是上面的还是下面的
        # 修改这里：处理候选新闻（正样本和负样本）
        # pos_news_id = line[4].split()  # 正样本新闻ID
        # neg_news_ids = line[5].split()  # 负样本新闻ID列表
        # candidate_news_ids = [pos_news_id] + neg_news_ids
        # candidate_index = self.trans_to_nindex(candidate_news_ids)
        # candidate_input = self.news_input[candidate_index]
        
        # # 检查candidate_input的值范围
        # if torch.isnan(candidate_input).any() or torch.isinf(candidate_input).any():
        #     print(f"Warning: Invalid values in candidate_input")
        #     candidate_input = torch.nan_to_num(candidate_input, nan=0.0, posinf=0.0, neginf=0.0)
        
        # # 确保所有值都是非负的
        # candidate_input = torch.clamp(candidate_input, min=0)
        
        # 修改这里：生成标签（正样本为1，负样本为0）
        # labels = np.array([1] + [0] * len(neg_news_ids))

        if self.cfg.model.use_entity:
            # 处理原始实体
            origin_entity = self.news_input[candidate_index]
            
            # 先不考虑情感向量
            # 获取候选新闻实体的情感向量
            # candidate_sentiment = []
            # for entities in origin_entity:
            #     entity_sentiments = []
            #     for entity_id in entities:
            #         if entity_id in self.entity_sentiment:
            #             entity_sentiments.append(self.entity_sentiment[entity_id])
            #         else:
            #             entity_sentiments.append(self.default_sentiment)
            #     candidate_sentiment.append(entity_sentiments)
            # candidate_sentiment = np.array(candidate_sentiment)
            
            # 处理邻居实体
            candidate_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, 
                 self.cfg.model.entity_neighbors), 
                dtype=np.int64
            )
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]
            
            candidate_neighbor_entity = candidate_neighbor_entity.reshape(
                len(candidate_index),
                self.cfg.model.entity_size * self.cfg.model.entity_neighbors
            )
            
            # 处理邻居实体的情感向量
            # neighbor_sentiment = np.zeros((len(candidate_index),
            #                                self.cfg.model.entity_size * self.cfg.model.entity_neighbors, 3))
            # for i, neighbors in enumerate(candidate_neighbor_entity):
            #     for j, neighbor_id in enumerate(neighbors):
            #         if neighbor_id in self.entity_sentiment:
            #             neighbor_sentiment[i, j] = self.entity_sentiment[neighbor_id]
            #         else:
            #             neighbor_sentiment[i, j] = self.default_sentiment
            
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            
            # 合并所有信息
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
            # candidate_sentiment = np.concatenate((candidate_sentiment, neighbor_sentiment), axis=-1)
            # candidate_entity = np.concatenate((candidate_entity, candidate_sentiment), axis=-1)
            
            # 合并点击新闻的实体和情感向量
            # clicked_entity = np.concatenate((clicked_entity, clicked_sentiment), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)
            # clicked_entity = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])
        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels
    
    @abstractmethod
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:  # 确保有点击历史
                result = self.line_mapper(line)
                (batch, mapping_idx, clicked_entity, candidate_input,
                 candidate_entity, entity_mask, labels) = result
                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels
            else:
                continue  # 如果没有点击历史，跳过这一行
