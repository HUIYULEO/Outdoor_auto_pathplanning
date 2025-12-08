import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import SegmentationDataset, BATCH_SIZE
from training.utils import save_checkpoint
from training import config as tconfig

# Import model architectures
try:
    from models.unet import UNet
    from models.unetpp import NestedUNet
    from models.attunet import AttU_Net
except ImportError:
    from models import UNet, NestedUNet, AttU_Net


def get_model(model_type, device):
    """Get model by type"""
    if model_type == 'unet':
        return UNet().to(device)
    elif model_type == 'unetpp':
        return NestedUNet().to(device)
    elif model_type == 'attunet':
        return AttU_Net().to(device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--data', required=True, help='Dataset root directory')
    parser.add_argument('--model_type', default='unet', choices=['unet', 'unetpp', 'attunet'])
    parser.add_argument('--epochs', type=int, default=tconfig.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=tconfig.LR)
    parser.add_argument('--checkpoint_dir', default=tconfig.CHECKPOINT_DIR)
    parser.add_argument('--device', default=tconfig.DEVICE)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create dataset and loader
    dataset = SegmentationDataset(
        os.path.join(args.data, 'images'),
        os.path.join(args.data, 'masks')
    )
    
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(0.1 * num_train)
    train_idx, val_idx = indices[split:], indices[:split]
    
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)

    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

    # Create model
    model = get_model(args.model_type, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        ckpt_name = f"{args.model_type}_epoch{epoch}.pth"
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss
        }, args.checkpoint_dir, filename=ckpt_name)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss
            }, args.checkpoint_dir, filename=f"{args.model_type}_best.pth")
    
    print("Training finished.")


if __name__ == '__main__':
    main()
