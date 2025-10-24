import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import time
import math
import random
import numpy as np
from tqdm import tqdm

from src.model import Transformer
from src.data import get_dataloaders, PAD_TOKEN
from src.utils import plot_curves

def set_seed(seed):
    """设置随机种子以保证可复现性 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx_src, pad_idx_tgt):
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for src, tgt in pbar:
        src = src.to(device) # (batch_size, src_len)
        tgt = tgt.to(device) # (batch_size, tgt_len)
        
        # 目标(tgt)需要拆分为:
        # tgt_input: (batch_size, tgt_len - 1) -> <bos> ... token_n
        # tgt_output: (batch_size, tgt_len - 1) -> token_1 ... <eos>
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        output = model(src, tgt_input, pad_idx_src, pad_idx_tgt)
        
        # output: (batch_size, tgt_len - 1, tgt_vocab_size)
        # tgt_output: (batch_size, tgt_len - 1)
        
        output_flat = output.contiguous().view(-1, output.shape[-1])
        tgt_output_flat = tgt_output.contiguous().view(-1)
        
        loss = criterion(output_flat, tgt_output_flat)
        
        loss.backward()
        
        # 梯度裁剪 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, pad_idx_src, pad_idx_tgt):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input, pad_idx_src, pad_idx_tgt)
            
            output_flat = output.contiguous().view(-1, output.shape[-1])
            tgt_output_flat = tgt_output.contiguous().view(-1)
            
            loss = criterion(output_flat, tgt_output_flat)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
    return epoch_loss / len(dataloader)

def main(args):
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_dl, valid_dl, vocab_transform, pad_idx_src, pad_idx_tgt = get_dataloaders(args.batch_size)
    
    src_vocab_size = len(vocab_transform['de'])
    tgt_vocab_size = len(vocab_transform['en'])
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(device)
    
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True) # 
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_tgt) # 忽略pad
    
    # 训练循环
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device, pad_idx_src, pad_idx_tgt)
        valid_loss = evaluate(model, valid_dl, criterion, device, pad_idx_src, pad_idx_tgt)
        
        end_time = time.time()
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        epoch_mins = (end_time - start_time) // 60
        epoch_secs = int(end_time - start_time) % 60
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
        # 学习率调度
        scheduler.step(valid_loss)
        
        # 保存最佳模型 
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Best model saved to {args.save_path}")

    # 保存训练曲线
    if args.plot_path:
        plot_curves(train_losses, valid_losses, args.plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer from Scratch')
    
    # 超参
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练设置
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42) # 
    
    # 路径
    parser.add_argument('--save_path', type=str, default='models/transformer.pt')
    parser.add_argument('--plot_path', type=str, default='results/train_loss_curve.png') # 
    
    args = parser.parse_args()
    main(args)