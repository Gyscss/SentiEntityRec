from torch.utils.data import DataLoader
from .dataset import TrainGraphDataset, ValidGraphDataset, TestGraphDataset
import torch

def get_dataloader(cfg, mode):
    """获取数据加载器"""
    if mode == 'train':
        dataset = TrainGraphDataset(cfg)
        if cfg.model.test_mode:
            # 测试模式下只使用少量数据
            dataset = torch.utils.data.Subset(dataset, range(min(cfg.model.test_batch_size * cfg.model.test_num_batches, len(dataset))))
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.model.test_batch_size if cfg.model.test_mode else cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True
        )
    elif mode == 'valid':
        dataset = ValidGraphDataset(cfg)
        if cfg.model.test_mode:
            dataset = torch.utils.data.Subset(dataset, range(min(cfg.model.test_batch_size * cfg.model.test_num_batches, len(dataset))))
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.model.test_batch_size if cfg.model.test_mode else cfg.valid.batch_size,
            shuffle=False,
            num_workers=cfg.valid.num_workers,
            pin_memory=True
        )
    elif mode == 'test':
        dataset = TestGraphDataset(cfg)
        if cfg.model.test_mode:
            dataset = torch.utils.data.Subset(dataset, range(min(cfg.model.test_batch_size * cfg.model.test_num_batches, len(dataset))))
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.model.test_batch_size if cfg.model.test_mode else cfg.test.batch_size,
            shuffle=False,
            num_workers=cfg.test.num_workers,
            pin_memory=True
        )
    else:
        raise ValueError(f"Mode {mode} not found")
    return dataloader