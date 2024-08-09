# -*- coding: utf-8 -*-
import argparse
from os.path import join

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from data.data_generator import RNADataset
from models.model import DiffusionRNA2dPrediction
from optim.scheduler import LinearWarmupScheduler
from trainer import Trainer
from models.rna_model.config import (
    TransformerConfig,
    OptimizerConfig,
    Config,
    DataConfig,
    ProduceConfig,
    TrainConfig,
    LoggingConfig,
)

# Set random seed for reproducibility
seed_everything(42)

# Constants
MODEL_PATH = "/home/fkli/RNAm"
DATA_PATH = "/home/fkli/RNAdata/bpRNA_lasted/data"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNA Diffusion Model Training")

    # Model parameters
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--diffusion_dim", type=int, default=8, help="Dimension of diffusion")
    parser.add_argument("--cond_dim", type=int, default=1, help="Conditioning dimension")
    parser.add_argument("--dp_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--u_conditioner_ckpt", type=str, default="ufold_train_alldata.pt", help="U conditioner checkpoint")
    parser.add_argument("--esm_conditioner_ckpt", type=str, default="RNA-ESM2-trans-2a100-mappro-KDNY-epoch_06-valid_F1_0.564.ckpt", help="ESM conditioner checkpoint")

    # Data parameters
    parser.add_argument("--upsampling", type=eval, default=False, help="Whether to use upsampling")

    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup epochs")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--momentum_sqr", type=float, default=0.999, help="Squared momentum")
    parser.add_argument("--milestones", type=eval, default=[], help="Milestones for learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--pin_memory", type=eval, default=False, help="Whether to pin memory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--parallel", type=str, default=None, choices={"dp"}, help="Parallelization strategy")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Perform a dry run")

    # Logging parameters
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--project", type=str, default=None, help="Project name")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluation frequency")
    parser.add_argument("--check_every", type=int, default=None, help="Checkpoint frequency")
    parser.add_argument("--log_tb", type=eval, default=True, help="Log to TensorBoard")
    parser.add_argument("--log_wandb", type=eval, default=True, help="Log to Weights & Biases")
    parser.add_argument("--log_home", type=str, default=None, help="Logging directory")

    return parser.parse_args()

def create_data_loaders(args):
    """Create data loaders for training, validation, and testing."""
    train_dataset = RNADataset([join(DATA_PATH, "train")], upsampling=args.upsampling)
    val_dataset = RNADataset([join(DATA_PATH, "val")])
    test_dataset = RNADataset([join(DATA_PATH, "test")])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=val_dataset.collate_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=test_dataset.collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def create_optimizer_and_schedulers(model, args):
    """Create optimizer and learning rate schedulers."""
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr)
    )

    scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup) if args.warmup else None
    scheduler_epoch = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma) if args.milestones else None

    return optimizer, scheduler_iter, scheduler_epoch

def main():
    args = parse_arguments()

    # Initialize model
    model = DiffusionRNA2dPrediction(
        num_classes=args.num_classes,
        diffusion_dim=args.diffusion_dim,
        cond_dim=args.cond_dim,
        diffusion_steps=args.diffusion_steps,
        dropout_rate=args.dp_rate,
        unet_checkpoint=args.u_conditioner_ckpt,
        esm_checkpoint=args.esm_conditioner_ckpt,
        device=args.device,
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Create optimizer and schedulers
    optimizer, scheduler_iter, scheduler_epoch = create_optimizer_and_schedulers(model, args)

    # Initialize trainer
    trainer = Trainer(
        args=args,
        data_id="bpRNA",
        model_id="DiT",
        optim_id="adam",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler_iter=scheduler_iter,
        scheduler_epoch=scheduler_epoch,
    )

    # Start training
    trainer.run()

if __name__ == "__main__":
    main()