import os
import torch

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path

def load_checkpoint(path, model=None, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    if model is not None and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint
