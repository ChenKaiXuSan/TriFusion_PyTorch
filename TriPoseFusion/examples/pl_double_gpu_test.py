#!/usr/bin/env python3
"""
PyTorch Lightning 双 GPU 测试脚本

用法:
  python3 code/pl_double_gpu_test.py --gpus 2 --batch_size 64

要求:
  - Python 3.8+
  - torch, pytorch-lightning 已安装
"""
import os
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
# os.environ.setdefault('NCCL_SHM_DISABLE', '1')

import argparse
import multiprocessing as mp
import time

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import signal


def _cpu_worker(duration_seconds: int):
    # busy loop to consume CPU
    end = time.time() + duration_seconds
    x = 0
    while time.time() < end:
        # simple arithmetic to keep CPU busy
        for i in range(10000):
            x += (i * i) % (i + 1)


def _mem_worker(megabytes: int, duration_seconds: int):
    # allocate roughly `megabytes` MB and touch pages
    try:
        block_size = 1024 * 1024  # 1 MB
        blocks = []
        for _ in range(megabytes):
            blocks.append(bytearray(block_size))
        # touch memory periodically
        end = time.time() + duration_seconds
        while time.time() < end:
            for b in blocks:
                b[0] = (b[0] + 1) % 256
            time.sleep(0.5)
    except MemoryError:
        print(f"Memory worker: failed to allocate {megabytes} MB")


class ComplexStressDataset(Dataset):
    def __init__(self, size=1024000, dense_dim=32, vocab_size=4096, max_tokens=64, seed=1234):
        self.len = size
        self.dense_dim = dense_dim
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.seed = seed
        self.base_dense = torch.randn(size, dense_dim)
        self.lengths = torch.randint(low=8, high=max_tokens + 1, size=(size,))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)

        length = int(self.lengths[index].item())
        token_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(length,),
            generator=generator,
        )

        dense = self.base_dense[index].clone()
        dense_noise = torch.randn(
            dense.shape,
            generator=generator,
            device=dense.device,
            dtype=dense.dtype,
        )
        dense = dense + 0.05 * dense_noise
        dense = torch.tanh(dense)

        aux = torch.tensor(
            [float(index % 17), float(length), float(token_ids.float().mean().item()) / self.vocab_size],
            dtype=torch.float32,
        )

        # synthetic regression target with several interacting parts
        target = (
            dense.mean()
            + 0.01 * token_ids.float().mean()
            + 0.001 * token_ids.float().std(unbiased=False)
            + 0.02 * aux[0]
            + 0.01 * aux[1]
        )

        return {
            'token_ids': token_ids,
            'dense': dense,
            'aux': aux,
            'target': target.to(torch.float32),
            'index': torch.tensor(index, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
        }


def complex_collate_fn(batch):
    token_ids = [item['token_ids'] for item in batch]
    lengths = torch.tensor([item['length'].item() for item in batch], dtype=torch.long)
    padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=0)
    token_mask = torch.arange(padded_tokens.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    dense = torch.stack([item['dense'] for item in batch], dim=0)
    aux = torch.stack([item['aux'] for item in batch], dim=0)
    target = torch.stack([item['target'] for item in batch], dim=0)
    index = torch.stack([item['index'] for item in batch], dim=0)

    return {
        'token_ids': padded_tokens,
        'token_mask': token_mask,
        'dense': dense,
        'aux': aux,
        'target': target,
        'index': index,
        'lengths': lengths,
    }


class SimpleModel(pl.LightningModule):
    def __init__(self, input_dim=32, vocab_size=4096, token_dim=48):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, token_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.aux_proj = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(token_dim + 32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        token_ids = batch['token_ids']
        token_mask = batch['token_mask'].to(token_ids.device)
        dense = batch['dense']
        aux = batch['aux']

        token_emb = self.token_embed(token_ids)
        token_emb = self.token_proj(token_emb)
        mask = token_mask.unsqueeze(-1).to(token_emb.dtype)
        token_sum = (token_emb * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp_min(1.0)
        token_pooled = token_sum / token_count

        dense_feat = self.dense_proj(dense)
        aux_feat = self.aux_proj(aux)
        features = torch.cat([token_pooled, dense_feat, aux_feat], dim=-1)
        return self.head(features).squeeze(-1)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch['target'])
        # debug print to trace distributed progress
        try:
            rank = getattr(self, 'global_rank', None)
        except Exception:
            rank = None
        print(f"[test_step] rank={rank} batch={batch_idx} pid={os.getpid()} max_len={batch['token_ids'].shape[1]}")
        # 在分布式情况下同步日志
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def test_dataloader(self):
        # create dataloader inside each process to avoid pickling issues
        batch_size = getattr(self, 'batch_size', 64)
        num_workers = getattr(self, 'num_workers', 0)
        dataset = ComplexStressDataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=complex_collate_fn,
            pin_memory=True,
        )


def make_dataloader(batch_size: int):
    dataset = ComplexStressDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=complex_collate_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=2, help='请求使用的 GPU 数量')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--strategy', type=str, default=None, help='分布式策略，例如 ddp, ddp_spawn（默认: 自动选择）')
    parser.add_argument('--cpu_workers', type=int, default=0, help='启动多少个 CPU 压力进程（0=不启用）')
    parser.add_argument('--mem_mb', type=int, default=0, help='每个内存进程分配多少 MB（0=不启用）')
    parser.add_argument('--mem_workers', type=int, default=0, help='启动多少个内存压力进程（0=不启用）')
    parser.add_argument('--stress_duration', type=int, default=300, help='压力测试持续秒数')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 的 num_workers，默认0以避免与多进程冲突')
    args = parser.parse_args()

    # Ensure a safe multiprocessing start method for cross-platform and DDP
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # already set
        pass

    available = torch.cuda.device_count()
    print(f"Detected GPUs: {available}")
    if args.gpus > available:
        print(f"警告: 请求 {args.gpus} GPUs，但只检测到 {available}。将使用可用数量。")
        args.gpus = max(1, available)

    devices = args.gpus if args.gpus > 0 else 1
    accelerator = 'gpu' if args.gpus > 0 else 'cpu'
    strategy = 'ddp' if devices > 1 else 'auto'
    if args.strategy:
        strategy = args.strategy

    model = SimpleModel()
    # attach loader settings to model so each DDP process creates its own DataLoader
    model.batch_size = args.batch_size
    model.num_workers = args.num_workers

    stress_procs = []

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=100,
        inference_mode=False,
        barebones=True,
        logger=False,
        enable_checkpointing=False,
    )

    print(f"Starting test on accelerator={accelerator}, devices={devices}, strategy={strategy}")

    # 仅在全局主进程上启动压力进程，避免在每个 DDP 子进程重复创建
    master = False
    if hasattr(trainer, 'is_global_zero'):
        val = trainer.is_global_zero
        master = val() if callable(val) else bool(val)
    else:
        master = getattr(trainer, 'global_rank', 0) == 0

    if master:
        if args.cpu_workers > 0:
            print(f"[global_zero] Starting {args.cpu_workers} CPU worker(s) for {args.stress_duration}s")
            for _ in range(args.cpu_workers):
                p = mp.Process(target=_cpu_worker, args=(args.stress_duration,))
                p.start()
                stress_procs.append(p)

        if args.mem_workers > 0 and args.mem_mb > 0:
            print(f"[global_zero] Starting {args.mem_workers} memory worker(s), {args.mem_mb} MB each, for {args.stress_duration}s")
            for _ in range(args.mem_workers):
                p = mp.Process(target=_mem_worker, args=(args.mem_mb, args.stress_duration))
                p.start()
                stress_procs.append(p)

    try:
        trainer.test(model)
    finally:
        # 清理压力进程（只有主进程会创建）
        if stress_procs:
            print("Stopping stress processes...")
        for p in stress_procs:
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            except Exception:
                pass


if __name__ == '__main__':
    main()
