import os
import torch
import argparse

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from accelerate import Accelerator
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from vit import ViTv2
from optimization import train, validate


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_arg_parser():
    # Command Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--we', type=int, default=30, help='Number of warmup epochs')
    parser.add_argument('--mag', type=int, default=10, help='Magnitude of rand augment')
    parser.add_argument('--rs', type=int, default=7, help='Number of restarts')
    parser.add_argument('--epochs', type=int, default=160, help='Number of epochs')

    parser.add_argument('--layers', type=int, default=12, help='ViT Layers')
    parser.add_argument('--atts', type=int, default=384, help='Attention Layer Size')
    parser.add_argument('--ffs', type=int, default=1536, help='Feed Forwartd Layers Size')
    parser.add_argument('--heads', type=int, default=6, help='Number of Heads')
    return parser


def  main():
    # Create parser & fetch args
    parser = create_arg_parser()
    args = parser.parse_args()

    # Instantiate accelerator & Tracker
    accelerator = Accelerator()

    # Epoch Config.
    BATCH_SIZE = 512
    EPOCHS = args.epochs
    
    # Scheduler Config
    MIN_LR = 5e-7
    LEARNING_RATE = args.lr
    WARM_UP_EPOCHS = args.we
    RESTARTS = args.rs

    # Lambda scheduler for warmup
    def warmup_schedule(epoch): # I hate to define fucntions like this but I dont see the alternative
        if epoch < WARM_UP_EPOCHS:
            return (epoch+1) / (WARM_UP_EPOCHS+1)
        return 1.0

    # Input Config.
    resize_image_to = 64 + 8
    crop_to = 64
    patch_size = 8
    rand_aug_magnitude = args.mag

    # Transformer Config.
    heads = args.heads
    attn_size = args.atts
    ffn_size = args.ffs
    layers = args.layers
    n_clases = 200
   
    # ViT for cifar100
    vit_tiny = ViTv2(
        image_size = resize_image_to,
        patch_size = patch_size,
        num_classes = n_clases,
        att_dim = attn_size,
        depth = layers,
        heads = heads,
        mlp_dim = ffn_size,
        dropout = 0.0, 
        emb_dropout = 0.0
    )

    total_params = count_parameters(vit_tiny)
    print('Total Params', total_params)

    # CUDA Device
    device = accelerator.device
    vit_tiny.to(device)

    # Optimization Args
    optimizer = optim.Adam(vit_tiny.parameters(), lr=LEARNING_RATE)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(EPOCHS/RESTARTS), T_mult=2, eta_min=MIN_LR)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule)
    global_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARM_UP_EPOCHS])

    criterion = nn.CrossEntropyLoss()

    # Data Transforms
    transform_train = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(resize_image_to),
        v2.RandomCrop(crop_to),
        v2.RandAugment(num_ops=2, magnitude=rand_aug_magnitude),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = v2.Compose([
        v2.PILToTensor(),
        v2.Resize(resize_image_to),
        v2.CenterCrop(crop_to),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Define the paths to the dataset
    dataset_dir = '/data/tiny-imagenet-200'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    # Ceate Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=int(BATCH_SIZE), shuffle=False, num_workers=6)

    vit_tiny, optimizer, train_loader, val_loader, global_scheduler = accelerator.prepare(
        vit_tiny, optimizer, train_loader, val_loader, global_scheduler
    )

    for epoch in range(EPOCHS):
        if accelerator.is_main_process:
            print()
            print('Epoch ', epoch, 'lr', global_scheduler.get_last_lr())
        
        _ = train(epoch, accelerator, vit_tiny, train_loader, criterion, optimizer, n_clases)
        _ = validate(epoch, accelerator, vit_tiny, val_loader, criterion, n_clases)
        
        global_scheduler.step()

    # Make sure that the wandb tracker finishes correctly
    accelerator.end_training()
        
        
if __name__ == "__main__":
    main()    