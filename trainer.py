# -*- coding: utf-8 -*-
import os
import sys
import time
import wandb
import pickle
import pathlib
import numpy as np
import pandas as pd
from os.path import join

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.dicts import clean_dict
from utils.tables import get_args_table, get_metric_table
from utils.data import contact_map_masks, decode_name
from utils.loss import bce_loss, evaluate_f1_precision_recall
from utils.loss import (
    calculate_auc,
    calculate_mattews_correlation_coefficient,
    rna_evaluation,
)


sys.path.append("/home/fkli/Projects/RNADiffusion")

HOME = str(pathlib.Path.home())


class EarlyStopping(object):
    """
    Early stopping to terminate training when validation performance stops improving.
    """

    def __init__(self, patience=5, min_delta=1e-3):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_f1_score, current_epoch):
        """
        Args:
            val_f1_score (float): The current validation F1 score.
            current_epoch (int): The current epoch number.
        """
        if val_f1_score > self.best_score + self.min_delta:
            self.best_score = val_f1_score
            self.best_epoch = current_epoch
            self.counter = 0
            self.ckpt_save = True
            self.save_name = f"best_checkpoint_{self.best_epoch + 1}.pt"

        else:
            self.counter += 1
            self.ckpt_save = False
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class BaseTrainer(object):
    """
    Base class for training models.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler_iter,
        scheduler_epoch,
        log_path,
        eval_every,
        check_every,
        save_name=None,
    ):
        # Model and optimization
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = join(log_path, "check")

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every
        self.save_name = save_name

        # Initialize
        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []
        self.test_metrics = {}

        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0)

    def train(self, epoch):
        raise NotImplementedError()

    def validation(self, epoch):
        raise NotImplementedError()

    def test(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_metrics, val_metrics, test_metrics):
        raise NotImplementedError()

    def log_train_metrics(self, train_metrics):
        self._log_metrics(self.train_metrics, train_metrics)

    def log_eval_metrics(self, eval_metrics):
        self._log_metrics(self.eval_metrics, eval_metrics)

    def log_test_metrics(self, test_metrics):
        if test_metrics:
            self._log_metrics(self.test_metrics, test_metrics)
        else:
            ValueError("test_metrics is empty!")

    def _log_metrics(self, metrics_dict, new_metrics):
        if not metrics_dict:
            metrics_dict.update(
                {
                    metric_name: [metric_value]
                    for metric_name, metric_value in new_metrics.items()
                }
            )
        else:
            for metric_name, metric_value in new_metrics.items():
                metrics_dict[metric_name].append(metric_value)

    def create_folders(self):
        os.makedirs(self.log_path, exist_ok=True)
        print("Storing logs in:", self.log_path)

        if self.check_every is not None:
            os.makedirs(self.check_path, exist_ok=True)
            print("Storing checkpoints in:", self.check_path)

    def save_args(self, args):

        # Save args
        with open(join(self.log_path, "args.pickle"), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(join(self.log_path, "args_table.txt"), "w") as f:
            f.write(str(args_table))

    def save_metrics(self):

        # Save metrics
        self._save_pickle(self.train_metrics, "metrics_train.pickle")
        self._save_pickle(self.eval_metrics, "metrics_eval.pickle")
        self._save_pickle(self.test_metrics, "metrics_test.pickle")

        # Save metrics table
        self._save_metrics_table(
            self.train_metrics,
            "metrics_train.txt",
            list(range(1, self.current_epoch + 2)),
        )
        self._save_metrics_table(
            self.eval_metrics, "metrics_eval.txt", [e + 1 for e in self.eval_epochs]
        )
        self._save_metrics_table(
            self.test_metrics, "metrics_test.txt", [self.current_epoch + 1]
        )

    def _save_pickle(self, data, filename):
        with open(join(self.log_path, filename), "wb") as f:
            pickle.dump(data, f)

    def _save_metrics_table(self, metrics, filename, epochs):
        metric_table = get_metric_table(metrics, epochs=epochs)
        with open(join(self.log_path, filename), "w") as f:
            f.write(str(metric_table))

    def save_checkpoint(self, name="checkpoint.pt"):
        checkpoint = {
            "current_epoch": self.current_epoch,
            "train_metrics": self.train_metrics,
            "eval_metrics": self.eval_metrics,
            "eval_epochs": self.eval_epochs,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler_iter": (
                self.scheduler_iter.state_dict() if self.scheduler_iter else None
            ),
            "scheduler_epoch": (
                self.scheduler_epoch.state_dict() if self.scheduler_epoch else None
            ),
        }
        torch.save(checkpoint, join(self.check_path, name))

    def load_checkpoint(self, check_path, name="checkpoint.pt"):
        checkpoint = torch.load(join(check_path, name))
        self.current_epoch = checkpoint["current_epoch"]
        self.train_metrics = checkpoint["train_metrics"]
        self.eval_metrics = checkpoint["eval_metrics"]
        self.eval_epochs = checkpoint["eval_epochs"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler_iter:
            self.scheduler_iter.load_state_dict(checkpoint["scheduler_iter"])
        if self.scheduler_epoch:
            self.scheduler_epoch.load_state_dict(checkpoint["scheduler_epoch"])

    def run(self, epochs):
        for epoch in range(self.current_epoch, epochs):

            # Train
            train_metrics = self.train(epoch)
            self.log_train_metrics(train_metrics)

            # validation
            if (epoch + 1) % self.eval_every == 0:
                val_metrics = self.validation(epoch)
                self.early_stopping(val_metrics["f1"], epoch)
                if self.early_stopping.ckpt_save:
                    self.save_checkpoint(name=self.early_stopping.save_name)
                self.log_eval_metrics(val_metrics)
                self.eval_epochs.append(epoch)
            else:
                val_metrics = None

            # test
            if (epoch + 1) == epochs:
                if self.early_stopping.save_name is not None:
                    self.load_checkpoint(
                        self.check_path, name=self.early_stopping.save_name
                    )
                    print(f"load best checkpoint:{self.early_stopping.save_name}")
                else:
                    print("load last checkpoint")
                val_metrics, f1_pre_rec_df = self.test(epoch)
                f1_pre_rec_df.to_csv(
                    join(self.log_path, f"{self.save_name}.csv"),
                    index=False,
                    header=False,
                )
                self.log_test_metrics(val_metrics)
            elif self.early_stopping.early_stop:
                print("Early stopping")
                if self.early_stopping.save_name is not None:
                    self.load_checkpoint(
                        self.check_path, name=self.early_stopping.save_name
                    )
                    print(f"load best checkpoint:{self.early_stopping.save_name}")
                else:
                    print("load last checkpoint")
                val_metrics, f1_pre_rec_df = self.test(epoch)
                f1_pre_rec_df.to_csv(
                    join(self.log_path, f"{self.save_name}.csv"),
                    index=False,
                    header=True,
                )
                self.log_test_metrics(val_metrics)
                # Log
                self.save_metrics()
                self.log_fn(epoch, train_metrics, val_metrics, val_metrics)
                break
            else:
                val_metrics = None

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_metrics, val_metrics, val_metrics)

            # Checkpoint
            self.current_epoch += 1
            if (epoch + 1) % self.check_every == 0:
                self.save_checkpoint()


class DiffusionTrainer(BaseTrainer):
    no_log_keys = [
        "project",
        "name",
        "log_tb",
        "log_wandb",
        "check_every",
        "eval_every",
        "device",
        "parallel",
        "pin_memory",
        "num_workers",
    ]

    def __init__(
        self,
        args,
        data_id,
        model_id,
        optim_id,
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        scheduler_iter,
        scheduler_epoch,
    ):
        self.log_base = join(args.log_home or HOME, "logs", "RNADiffusion")
        args.eval_every = args.eval_every or args.epochs
        args.check_every = args.check_every or args.epochs
        args.name = args.name or time.strftime("%Y-%m-%d_%H-%M-%S")
        args.project = args.project or "RNADiffusion"

        save_name = (
            f'{args.name}.bpRNA.seed_{args.seed}.{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        )

        super(DiffusionTrainer, self).__init__(
            model=model,
            optimizer=optimizer,
            scheduler_iter=scheduler_iter,
            scheduler_epoch=scheduler_epoch,
            log_path=join(self.log_base, f"{data_id}_{model_id}_{optim_id}", args.name),
            eval_every=args.eval_every,
            check_every=args.check_every,
            save_name=save_name,
        )

        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # dry run
        self.dry_run = args.dry_run

        # Init logging
        if not self.dry_run:
            args_dict = clean_dict(vars(args), keys=self.no_log_keys)
            if args.log_tb:
                self.writer = SummaryWriter(join(self.log_path, "tb"))
                self.writer.add_text(
                    "args", get_args_table(args_dict).get_html_string(), global_step=0
                )
            if args.log_wandb:
                wandb.init(
                    config=args_dict,
                    project=args.project,
                    id=args.name,
                    dir=self.log_path,
                )

    def log_metrics(self, epoch, train_metrics, val_metrics, test_metrics):
        if not self.dry_run:
            if self.args.log_tb:
                self._log_to_tensorboard(
                    epoch, train_metrics, val_metrics, test_metrics
                )
            if self.args.log_wandb:
                self._log_to_wandb(epoch, train_metrics, val_metrics, test_metrics)

    def _log_to_tensorboard(self, epoch, train_metrics, val_metrics, test_metrics):
        for metric_name, metric_value in train_metrics.items():
            self.writer.add_scalar(
                f"train/{metric_name}", metric_value, global_step=epoch + 1
            )
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(
                    f"val/{metric_name}", metric_value, global_step=epoch + 1
                )
        if test_metrics:
            for metric_name, metric_value in test_metrics.items():
                self.writer.add_scalar(
                    f"test/{metric_name}", metric_value, global_step=epoch + 1
                )

    def _log_to_wandb(self, epoch, train_metrics, val_metrics, test_metrics):
        for metric_name, metric_value in train_metrics.items():
            wandb.log({f"train/{metric_name}": metric_value}, step=epoch + 1)
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                wandb.log({f"val/{metric_name}": metric_value}, step=epoch + 1)
        if test_metrics:
            metric_name_list = []
            metric_value_list = []
            for metric_name, metric_value in test_metrics.items():
                metric_name_list.append(metric_name)
                metric_value_list.append(metric_value)
                table = wandb.Table(columns=metric_name_list, data=[metric_value_list])
                wandb.log({"test": table})

    def resume(self):
        resume_path = join(
            self.log_base,
            f"{self.data_id}_{self.model_id}_{self.optim_id}",
            self.args.resume,
            "check",
        )
        self.load_checkpoint(resume_path)
        for epoch in range(self.current_epoch):
            train_metrics = {
                metric_name: metric_values[epoch]
                for metric_name, metric_values in self.train_metrics.items()
            }
            val_metrics = (
                {
                    metric_name: metric_values[self.eval_epochs.index(epoch)]
                    for metric_name, metric_values in self.eval_metrics.items()
                }
                if epoch in self.eval_epochs
                else None
            )
            test_metrics = (
                {
                    metric_name: metric_values[epoch]
                    for metric_name, metric_values in self.test_metrics.items()
                }
                if (epoch + 1) == self.args.epochs
                else None
            )
            self.log_metrics(
                epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )

    def run(self):
        if self.args.resume:
            self.resume()
        super().run(epochs=self.args.epochs)


class Trainer(DiffusionTrainer):

    def train(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        device = self.args.device

        for _, (
            _,
            raw_sequence,
            sequence_length,
            set_max_len,
            contact_map,
            base_info,
            sequence_encoding,
        ) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            contact_map = contact_map.squeeze(0).to(device)
            base_info = base_info.squeeze(0).to(device)
            sequence_length = sequence_length.squeeze(0).to(device)
            sequence_encoding = sequence_encoding.squeeze(0).to(device)
            matrix_rep = torch.zeros_like(contact_map)
            contact_masks = contact_map_masks(sequence_length, matrix_rep).to(device)

            loss = self.model(
                contact_map,
                base_info,
                raw_sequence,
                contact_masks,
                set_max_len,
                sequence_encoding,
            )
            loss.backward()
            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()

            total_loss += loss.detach().cpu().item() * len(contact_map)
            total_samples += len(contact_map)
            print(
                f"Training. Epoch: {epoch + 1}/{self.args.epochs}, Bits/dim: {total_loss / total_samples:.5f}\n"
            )

        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        return {"bpd": total_loss / total_samples}

    def validation(self, epoch):
        self.model.eval()
        device = self.args.device
        total_loss = 0.0
        total_samples = 0
        auc_score = 0.0
        auc_count = 0
        f1_scores = []
        mcc_scores = []
        with torch.no_grad():
            for _, (
                _,
                raw_sequence,
                sequence_length,
                set_max_len,
                contact_map,
                base_info,
                sequence_encoding,
            ) in enumerate(self.val_loader):
                contact_map = contact_map.squeeze(0)
                base_info = base_info.squeeze(0).to(device)
                sequence_length = sequence_length.squeeze(0).to(device)
                sequence_encoding = sequence_encoding.squeeze(0).to(device)
                matrix_rep = torch.zeros_like(contact_map)
                contact_masks = contact_map_masks(sequence_length, matrix_rep).to(
                    device
                )

                # calculate contact loss
                batch_size = contact_map.shape[0]
                pred_x0, _ = self.model.sample(
                    batch_size,
                    base_info,
                    raw_sequence,
                    set_max_len,
                    contact_masks,
                    sequence_encoding,
                )

                pred_x0 = pred_x0.cpu().float()
                total_loss += (
                    bce_loss(pred_x0.float(), contact_map.float()).cpu().item()
                )
                total_samples += len(contact_map)
                auc_score += calculate_auc(contact_map.float(), pred_x0)
                auc_count += 1

                f1_scores.extend(
                    evaluate_f1_precision_recall(
                        pred_x0[i].squeeze(), contact_map.float()[i].squeeze()
                    )
                    for i in range(pred_x0.shape[0])
                )

                mcc_scores.extend(
                    calculate_mattews_correlation_coefficient(
                        pred_x0[i].squeeze(), contact_map.float()[i].squeeze()
                    )
                    for i in range(pred_x0.shape[0])
                )

            val_precision, val_recall, val_f1 = zip(*f1_scores)
            val_precision = np.average(np.nan_to_num(np.array(val_precision)))
            val_recall = np.average(np.nan_to_num(np.array(val_recall)))
            val_f1 = np.average(np.nan_to_num(np.array(val_f1)))
            mcc_final = np.average(np.nan_to_num(np.array(mcc_scores)))

            print("#" * 80)
            print("Average val F1 score: ", round(val_f1, 3))
            print("Average val precision: ", round(val_precision, 3))
            print("Average val recall: ", round(val_recall, 3))
            print("Average val MCC", round(mcc_final, 3))
            print("#" * 80)

        return {
            "f1": val_f1,
            "precision": val_precision,
            "recall": val_recall,
            "auc_score": auc_score / auc_count,
            "mcc": mcc_final,
            "bce_loss": total_loss / total_samples,
        }

    def test(self, epoch):
        self.model.eval()
        device = self.args.device
        test_results = []
        total_name_list = []
        total_length_list = []

        with torch.no_grad():
            for _, (
                data_name,
                raw_sequence,
                sequence_length,
                set_max_len,
                contact_map,
                base_info,
                sequence_encoding,
            ) in enumerate(self.test_loader):
                sequence_length = sequence_length.squeeze(0).to(device)
                data_name_list = [
                    list(filter(lambda x: x != -1, item.numpy()))
                    for item in data_name.squeeze(0)
                ]
                total_name_list += [decode_name(item) for item in data_name_list]
                total_length_list += [item.item() for item in sequence_length]

                contact_map = contact_map.squeeze(0)
                base_info = base_info.squeeze(0).to(device)
                data_seq_encoding = data_seq_encoding.squeeze(0).to(device)
                matrix_rep = torch.zeros_like(contact_map)
                contact_masks = contact_map_masks(sequence_length, matrix_rep).to(
                    device
                )

                # calculate contact loss
                batch_size = contact_map.shape[0]
                pred_x0, _ = self.model.sample(
                    batch_size,
                    base_info,
                    raw_sequence,
                    set_max_len,
                    contact_masks,
                    sequence_encoding,
                )

                pred_x0 = pred_x0.cpu().float()

                test_results.extend(
                    rna_evaluation(
                        pred_x0[i].squeeze(), contact_map.float()[i].squeeze()
                    )
                    for i in range(pred_x0.shape[0])
                )

            accuracy, precision, recall, sens, spec, f1, mcc = zip(*test_results)

            results_df = pd.DataFrame(
                {
                    "name": total_name_list,
                    "length": total_length_list,
                    "accuracy": list(np.array(accuracy)),
                    "precision": list(np.array(precison)),
                    "recall": list(np.array(recall)),
                    "sensitivity": list(np.array(sens)),
                    "specificity": list(np.array(spec)),
                    "f1": list(np.array(f1)),
                    "mcc": list(np.array(mcc)),
                }
            )

            accuracy = np.average(np.nan_to_num(np.array(accuracy)))
            precision = np.average(np.nan_to_num(np.array(precision)))
            recall = np.average(np.nan_to_num(np.array(recall)))
            sensitivity = np.average(np.nan_to_num(np.array(sens)))
            specificity = np.average(np.nan_to_num(np.array(spec)))
            f1 = np.average(np.nan_to_num(np.array(f1)))
            mcc = np.average(np.nan_to_num(np.array(mcc)))

            print("#" * 40)
            print("Average testing accuracy: ", round(accuracy, 3))
            print("Average testing F1 score: ", round(f1, 3))
            print("Average testing precision: ", round(precision, 3))
            print("Average testing recall: ", round(recall, 3))
            print("Average testing sensitivity: ", round(sensitivity, 3))
            print("Average testing specificity: ", round(specificity, 3))
            print("Average testing MCC", round(mcc, 3))
            print("#" * 40)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            "mcc": mcc,
        }, results_df
