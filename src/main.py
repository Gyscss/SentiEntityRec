import os.path
from pathlib import Path

import hydra
import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import torch.nn as nn

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from utils.metrics import *
from utils.common import *


# 添加调试环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# 我添加的代码
torch.cuda.set_per_process_memory_fraction(0.8, device=0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

# custom your wandb setting here #
# os.environ["WANDB_API_KEY"] = ""
# 这里
os.environ["WANDB_MODE"] = "offline"



def train(model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping):
    """训练模型的函数
    
    Args:
        model: 要训练的模型
        optimizer: 优化器
        scaler: 梯度缩放器
        scheduler: 学习率调度器
        dataloader: 数据加载器
        local_rank: 本地设备编号
        cfg: 配置对象
        early_stopping: 早停对象
    """
    model.train()
    torch.set_grad_enabled(True)

    sum_loss = torch.zeros(1).to(local_rank)
    sum_auc = torch.zeros(1).to(local_rank)

    for cnt, (subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels) \
            in enumerate(tqdm(dataloader,
                              total=int(cfg.num_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)),
                              desc=f"[{local_rank}] Training"), start=1):
        subgraph = subgraph.to(local_rank, non_blocking=True)
        mapping_idx = mapping_idx.to(local_rank, non_blocking=True)
        candidate_news = candidate_news.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)
        candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
        entity_mask = entity_mask.to(local_rank, non_blocking=True)

        with amp.autocast():
            bz_loss, y_hat = model(subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, labels)
            
        # Accumulate the gradients
        scaler.scale(bz_loss).backward()
        if cnt % cfg.accumulation_steps == 0 or cnt == int(cfg.dataset.pos_count / cfg.batch_size):
            # Update the parameters
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                scheduler.step()
                # https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814
            optimizer.zero_grad(set_to_none=True)

        sum_loss += bz_loss.data.float()
        sum_auc += area_under_curve(labels, y_hat)
        # ---------------------------------------- Training Log
        if cnt % cfg.log_steps == 0:
            if local_rank == 0:
                wandb.log({"train_loss": sum_loss.item() / cfg.log_steps, "train_auc": sum_auc.item() / cfg.log_steps})
            print('[{}] Ed: {}, average_loss: {:.5f}, average_acc: {:.5f}'.format(
                local_rank, cnt * cfg.batch_size, sum_loss.item() / cfg.log_steps, sum_auc.item() / cfg.log_steps))
            sum_loss.zero_()
            sum_auc.zero_()

        if cnt > int(cfg.val_skip_epochs * (cfg.dataset.pos_count // cfg.batch_size + 1)) and cnt % cfg.val_steps == 0:
        # 为了测试val函数
        # if cnt == 1:
            res = val(model, local_rank, cfg)
            model.train()

            if local_rank == 0:
                pretty_print(res)
                wandb.log(res)

            early_stop, get_better = early_stopping(res['auc'])
            if early_stop:
                print("Early Stop.")
                break
            elif get_better:
                print(f"Better Result!")
                if local_rank == 0:
                    save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc{res['auc']}")
                    wandb.run.summary.update({"best_auc": res["auc"], "best_mrr": res['mrr'],
                                              "best_ndcg5": res['ndcg5'], "best_ndcg10": res['ndcg10']})

def val(model, local_rank, cfg):
    """验证模型的函数
        Args:
        model: 要验证的模型
        local_rank: 本地设备编号
        cfg: 配置对象
        
    Returns:
        dict: 包含验证指标的字典
    """
    model.eval()
    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank)
    tasks = []
    with torch.no_grad():
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels) \
                in enumerate(tqdm(dataloader, total=int(cfg.dataset.val_len / cfg.gpu_num), desc=f"[{local_rank}] Validating")):
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)

            scores = model.module.validation_process(
                subgraph, mappings, clicked_entity, 
                candidate_emb, candidate_entity, entity_mask
            )
            
            tasks.append((labels.tolist(), scores))

    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T

    # barrier
    torch.distributed.barrier()

    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }
    
    return res


def main_worker(local_rank, cfg):
    # # 设置分布式训练环境
    # if cfg.gpu_num > 1:
    #     # 使用本地主机地址
    #     os.environ['MASTER_ADDR'] = '127.0.0.1'
    #     os.environ['MASTER_PORT'] = '23456'
    #     dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.gpu_num, rank=local_rank)
    #     torch.cuda.set_device(local_rank)
    
    # # 设置随机种子
    # torch.manual_seed(cfg.seed)
    # np.random.seed(cfg.seed)
    # random.seed(cfg.seed)
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)

    # 在Linux 下，默认后端可能是 nccl 或 gloo。
    # dist.init_process_group(backend='nccl',
    #                         init_method='tcp://127.0.0.1:23456',
    #                         world_size=cfg.gpu_num,
    #                         rank=local_rank)
    # 在 Windows 下，nccl 不支持，必须使用 gloo 后端。
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23457'
    dist.init_process_group(backend='gloo',
                            init_method='env://',
                            world_size=cfg.gpu_num,
                            rank=local_rank)

    # -----------------------------------------Dataset & Model Load
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)

    # 加载数据
    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank)
    model = load_model(cfg).to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    def get_lr_multiplier(step):
        return 1.0 if step > num_warmup_steps else step / num_warmup_steps

    scheduler = LambdaLR(optimizer, get_lr_multiplier)
    
    # ------------------------------------------Load Checkpoint & optimizer
    if cfg.load_checkpoint:
        file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}.pth")
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # After Distributed
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer.zero_grad(set_to_none=True)
    scaler = amp.GradScaler()

    # ------------------------------------------Main Start
    early_stopping = EarlyStopping(cfg.early_stop_patience)

    if local_rank == 0:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                   project=cfg.logger.exp_name, name=cfg.logger.run_name)
        print(model)

    # for _ in tqdm(range(1, cfg.num_epochs + 1), desc="Epoch"):
    train(model, optimizer, scaler, scheduler, train_dataloader, local_rank, cfg, early_stopping)

    if local_rank == 0:
        wandb.finish()


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    print(cfg.seed)
    cfg.gpu_num = torch.cuda.device_count()
    print(cfg.gpu_num)
    prepare_preprocessed_data(cfg)
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))


if __name__ == "__main__":
    main()

# 在Ternimal里运行
# 切换到wsl环境
# 还要指定python编码器
# /mnt/c/Users/Lenovo/AppData/Local/Programs/Python/Python39/python.exe src/main.py model=SentiEntityRec dataset=MINDsmall reprocess=True
