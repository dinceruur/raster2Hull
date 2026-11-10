from mpl_toolkits.mplot3d import Axes3D
import math
import os
import torch
from torchview import draw_graph
import matplotlib.pyplot as plt


def save_checkpoint(state, checkpoint_dir="./", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, filepath)
    print(f"--> Saved checkpoint: {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    print(f"--> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    start_epoch = checkpoint['epoch']
    current_loss = checkpoint['train_loss']
    best_recon_loss = checkpoint.get('best_recon_loss', float('inf'))
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

    print(f"--> Loaded successfully. Resuming from epoch {start_epoch}")
    return start_epoch, current_loss, best_recon_loss, epochs_no_improve

def count_params(module, trainable_only=True):
    if trainable_only:
        
        print(f"\n{'='*40}")
        print(f"{'Module':<20} | {'Trainable Params':<15}")
        print(f"{'-'*20}-+-{'-'*15}")
        
        total_params = 0
        for name, child in module.named_children()
            count = sum(p.numel() for p in child.parameters() if p.requires_grad)
            print(f"{name:<20} | {count:>13,}")
            total_params += count
            
        print(f"{'-'*20}-+-{'-'*15}")
        print(f"{'TOTAL':<20} | {total_params:>13,}")
        print(f"{'='*40}\n")

        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def farthest_point_sampling(xyz, npoint):
    device = xyz.device
    N, _ = xyz.shape

    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), device=device)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance)

    return centroids