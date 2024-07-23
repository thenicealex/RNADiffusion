# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from os.path import join
import argparse
from data.data_generator import RNADataset, diff_collate_fn, get_data_id
from functools import partial

from models.model import DiffusionRNA2dPrediction, get_model_id
from optim.scheduler import LinearWarmupScheduler, get_optim_id
from trainer import Trainer
from pytorch_lightning import seed_everything
from models.rna_model.config import TransformerConfig, OptimizerConfig, Config, DataConfig, ProduceConfig, TrainConfig, LoggingConfig

seed_everything(42)

MODEL_PATH = "/home/fkli/RNAm"
DATA_PATH = "/home/fkli/Projects/DiffRNA/datasets/temp"


def setup_args():
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--diffusion_dim", type=int, default=8)
    parser.add_argument("--cond_dim", type=int, default=1)
    parser.add_argument("--dp_rate", type=float, default=0.0)
    parser.add_argument(
        "--esm_conditioner_ckpt",
        type=str,
        default="RNA-ESM2-trans-2a100-mappro-KDNY-epoch_06-valid_F1_0.564.ckpt",
    )

    # Data params
    parser.add_argument("--dataset", type=str, default="bpRNAnew")
    parser.add_argument(
        "--seq_len", type=str, default="160", choices={"160", "600", "640", "all"}
    )
    parser.add_argument("--upsampling", type=eval, default=False)

    # Optim params
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--momentum_sqr", type=float, default=0.999)
    parser.add_argument("--milestones", type=eval, default=[])
    parser.add_argument("--gamma", type=float, default=0.1)

    # Train params
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=eval, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--parallel", type=str, default=None, choices={"dp"})
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", default=False)

    # Logging params
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--check_every", type=int, default=None)
    parser.add_argument("--log_tb", type=eval, default=True)
    parser.add_argument("--log_wandb", type=eval, default=True)
    parser.add_argument("--log_home", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":

    args = setup_args()

    # model
    model_id = get_model_id(args)
    model = DiffusionRNA2dPrediction(
        num_classes=args.num_classes,
        diffusion_dim=args.diffusion_dim,
        cond_dim=args.cond_dim,
        diffusion_steps=args.diffusion_steps,
        dp_rate=args.dp_rate,
        esm_ckpt=args.esm_conditioner_ckpt,
    )
    alphabet = model.get_alphabet()

    # data
    data_id = get_data_id(args)
    train = RNADataset([join(DATA_PATH, "train")], upsampling=False)
    val = RNADataset([join(DATA_PATH, "val")])
    test = RNADataset([join(DATA_PATH, "test")])
    partial_collate_fn = partial(diff_collate_fn)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    # optimizer
    optim_id = get_optim_id(args)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr)
    )
    if args.warmup is not None:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None

    if len(args.milestones) > 0:
        scheduler_epoch = MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma
        )
    else:
        scheduler_epoch = None

    t = Trainer(
        args=args,
        data_id=data_id,
        model_id=model_id,
        optim_id=optim_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler_iter=scheduler_iter,
        scheduler_epoch=scheduler_epoch,
    )

    t.run()
