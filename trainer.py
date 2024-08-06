# -*- coding: utf-8 -*-
import os
import torch
import time
import pickle
import wandb
import pathlib
import numpy as np
import pandas as pd

from common.data_utils import decode_name, contact_map_masks
from common.loss_utils import bce_loss, evaluate_f1_precision_recall
from common.loss_utils import (
    calculate_auc,
    calculate_mattews_correlation_coefficient,
    rna_evaluation,
)
from os.path import join
from common.utils import get_args_table, get_metric_table, clean_dict

from torch.utils.tensorboard import SummaryWriter

HOME = str(pathlib.Path.home())


class EarlyStopping(object):
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = 0.0
        self.best_epoch = None
        self.counter = 0
        self.save_name = None
        self.early_stop = False
        self.ckpt_save = False

    def __call__(self, val_f1_score, current_epoch):
        if val_f1_score > self.best_score:
            self.best_score = val_f1_score
            self.best_epoch = current_epoch
            self.ckpt_save = True
            self.save_name = f"best_checkpoint_{self.best_epoch + 1}.pt"
            self.counter = 0

        elif val_f1_score - self.best_score < self.min_delta:
            self.ckpt_save = False
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class BaseTrainer(object):

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

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, "check")

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

    def log_fn(self, epoch, train_dict, val_dict, test_dict):
        raise NotImplementedError()

    def log_train_metrics(self, train_dict):
        if len(self.train_metrics) == 0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics) == 0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def log_test_metrics(self, test_dict):
        if test_dict is not None:
            if len(self.test_metrics) == 0:
                for metric_name, metric_value in test_dict.items():
                    self.test_metrics[metric_name] = [metric_value]
            else:
                for metric_name, metric_value in test_dict.items():
                    self.test_metrics[metric_name].append(metric_value)
        else:
            print("test_dict is empty!")

    def create_folders(self):
        # Create log folder
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None and not os.path.exists(self.check_path):
            os.makedirs(self.check_path)
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
        with open(join(self.log_path, "metrics_train.pickle"), "wb") as f:
            pickle.dump(self.train_metrics, f)
        with open(join(self.log_path, "metrics_eval.pickle"), "wb") as f:
            pickle.dump(self.eval_metrics, f)
        with open(join(self.log_path, "metrics_test.pickle"), "wb") as f:
            pickle.dump(self.test_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(
            self.train_metrics, epochs=list(range(1, self.current_epoch + 2))
        )
        with open(join(self.log_path, "metrics_train.txt"), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(
            self.eval_metrics, epochs=[e + 1 for e in self.eval_epochs]
        )
        with open(join(self.log_path, "metrics_eval.txt"), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(
            self.test_metrics, epochs=[self.current_epoch + 1]
        )
        with open(join(self.log_path, "metrics_test.txt"), "w") as f:
            f.write(str(metric_table))

    def checkpoint_save(self, name="checkpoint.pt"):
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
        torch.save(checkpoint, os.path.join(self.check_path, name))

    def checkpoint_load(self, check_path, name="checkpoint.pt"):
        checkpoint = torch.load(os.path.join(check_path, name))
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
            train_dict = self.train(epoch)
            self.log_train_metrics(train_dict)

            # val
            if (epoch + 1) % self.eval_every == 0:
                val_dict = self.validation(epoch)
                self.early_stopping(val_dict["f1"], epoch)
                if self.early_stopping.ckpt_save:
                    self.checkpoint_save(name=self.early_stopping.save_name)
                self.log_eval_metrics(val_dict)
                self.eval_epochs.append(epoch)
            else:
                val_dict = None

            # test
            if (epoch + 1) == epochs:
                if self.early_stopping.save_name is not None:
                    self.checkpoint_load(
                        self.check_path, name=self.early_stopping.save_name
                    )
                    print(f"load best checkpoint:{self.early_stopping.save_name}")
                else:
                    print("load last checkpoint")
                test_dict, f1_pre_rec_df = self.test(epoch)
                f1_pre_rec_df.to_csv(
                    join(self.log_path, f"{self.save_name}.csv"),
                    index=False,
                    header=False,
                )
                self.log_test_metrics(test_dict)
            elif self.early_stopping.early_stop:
                print("Early stopping")
                if self.early_stopping.save_name is not None:
                    self.checkpoint_load(
                        self.check_path, name=self.early_stopping.save_name
                    )
                    print(f"load best checkpoint:{self.early_stopping.save_name}")
                else:
                    print("load last checkpoint")
                test_dict, f1_pre_rec_df = self.test(epoch)
                f1_pre_rec_df.to_csv(
                    join(self.log_path, f"{self.save_name}.csv"),
                    index=False,
                    header=True,
                )
                self.log_test_metrics(test_dict)
                # Log
                self.save_metrics()
                self.log_fn(epoch, train_dict, val_dict, test_dict)
                break
            else:
                test_dict = None

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, val_dict, test_dict)

            # Checkpoint
            self.current_epoch += 1
            if (epoch + 1) % self.check_every == 0:
                self.checkpoint_save()


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

        if args.log_home is None:
            self.log_base = join(HOME, "logs", "RNADiffusion")
        else:
            self.log_base = join(args.log_home, "logs", "RNADiffusion")

        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.project is None:
            args.project = "RNADiffusion"

        save_name = f'{args.name}.{args.dataset}.seed_{args.seed}.{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        model.to(args.device)

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
                self.writer = SummaryWriter(os.path.join(self.log_path, "tb"))
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

    def log_fn(self, epoch, train_dict, val_dict, test_dict):

        if not self.dry_run:
            # Tensorboard
            if self.args.log_tb:
                for metric_name, metric_value in train_dict.items():
                    self.writer.add_scalar(
                        "train/{}".format(metric_name),
                        metric_value,
                        global_step=epoch + 1,
                    )

                if val_dict:
                    for metric_name, metric_value in val_dict.items():
                        self.writer.add_scalar(
                            "val/{}".format(metric_name),
                            metric_value,
                            global_step=epoch + 1,
                        )

                if test_dict:
                    for metric_name, metric_value in test_dict.items():
                        self.writer.add_scalar(
                            "test/{}".format(metric_name),
                            metric_value,
                            global_step=epoch + 1,
                        )

            # Weights & Biases
            if self.args.log_wandb:
                for metric_name, metric_value in train_dict.items():
                    wandb.log(
                        {"train/{}".format(metric_name): metric_value}, step=epoch + 1
                    )
                if val_dict:
                    for metric_name, metric_value in val_dict.items():
                        wandb.log(
                            {"val/{}".format(metric_name): metric_value}, step=epoch + 1
                        )
                if test_dict:
                    metric_name_list = []
                    metric_value_list = []
                    for metric_name, metric_value in test_dict.items():
                        metric_name_list.append(metric_name)
                        metric_value_list.append(metric_value)
                        table = wandb.Table(
                            columns=metric_name_list, data=[metric_value_list]
                        )
                        wandb.log({"test": table})

    def resume(self):
        resume_path = os.path.join(
            self.log_base,
            f"{self.data_id}_{self.model_id}_{self.optim_id}",
            self.args.resume,
            "check",
        )
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]

            if epoch in self.eval_epochs:
                val_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    val_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else:
                val_dict = None

            if (epoch + 1) == self.args.epochs:
                test_dict = {}
                for metric_name, metric_values in self.test_metrics.items():
                    test_dict[metric_name] = metric_values[epoch]
            else:
                test_dict = None

            self.log_fn(
                epoch, train_dict=train_dict, val_dict=val_dict, test_dict=test_dict
            )

    def run(self):
        if self.args.resume:
            self.resume()
        super(DiffusionTrainer, self).run(epochs=self.args.epochs)


class Trainer(DiffusionTrainer):

    def train(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        device = self.args.device
        for _, (
            set_max_len,
            _,
            data_length,
            contact,
            data_seq_raw,
            data_seq_encoding,
        ) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            contact = contact.squeeze(0).to(device)
            matrix_rep = torch.zeros_like(contact)
            data_seq_encoding = data_seq_encoding.squeeze(0).to(device)
            data_length = data_length.squeeze(0).to(device)
            # data_seq_raw = data_seq_raw.to(device)
            contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

            loss = self.model(
                contact, data_seq_raw, contact_masks, set_max_len, data_seq_encoding
            )
            loss.backward()

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(contact)
            loss_count += len(contact)
            print(
                "Training. Epoch: {}/{}, Bits/dim: {:.5f}".format(
                    epoch + 1, self.args.epochs, loss_sum / loss_count
                ),
                end="\r",
            )

        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        return {"bpd": loss_sum / loss_count}

    def validation(self, epoch):
        self.model.eval()

        device = self.args.device
        with torch.no_grad():
            loss_count = 0
            val_loss_sum = 0.0
            auc_score = 0.0
            auc_count = 0
            val_no_train = list()
            mcc_no_train = list()

            for _, (
                set_max_len,
                _,
                data_length,
                contact,
                data_seq_raw,
                data_seq_encoding,
            ) in enumerate(self.val_loader):
                contact = contact.squeeze(0)
                matrix_rep = torch.zeros_like(contact)
                data_seq_encoding = data_seq_encoding.squeeze(0).to(device)
                data_length = data_length.squeeze(0).to(device)
                contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                num_samples = contact.shape[0]
                pred_x0, _ = self.model.sample(
                    num_samples,
                    data_seq_raw,
                    set_max_len,
                    contact_masks,
                    data_seq_encoding,
                )

                pred_x0 = pred_x0.cpu().float()
                val_loss_sum += bce_loss(pred_x0.float(), contact.float()).cpu().item()
                loss_count += len(contact)
                auc_score += calculate_auc(contact.float(), pred_x0)
                auc_count += 1
                val_no_train_tmp = list(
                    map(
                        lambda i: evaluate_f1_precision_recall(
                            pred_x0[i].squeeze(), contact.float()[i].squeeze()
                        ),
                        range(pred_x0.shape[0]),
                    )
                )
                val_no_train += val_no_train_tmp

                mcc_no_train_tmp = list(
                    map(
                        lambda i: calculate_mattews_correlation_coefficient(
                            pred_x0[i].squeeze(), contact.float()[i].squeeze()
                        ),
                        range(pred_x0.shape[0]),
                    )
                )
                mcc_no_train += mcc_no_train_tmp

            val_precision, val_recall, val_f1 = zip(*val_no_train)

            val_precision = np.average(np.nan_to_num(np.array(val_precision)))
            val_recall = np.average(np.nan_to_num(np.array(val_recall)))
            val_f1 = np.average(np.nan_to_num(np.array(val_f1)))

            mcc_final = np.average(np.nan_to_num(np.array(mcc_no_train)))

            print("#" * 80)
            print("Average val F1 score: ", round(val_f1, 3))
            print("Average val precision: ", round(val_precision, 3))
            print("Average val recall: ", round(val_recall, 3))
            print("#" * 80)
            print("Average val MCC", round(mcc_final, 3))
            print("#" * 80)
            print("")
        return {
            "f1": val_f1,
            "precision": val_precision,
            "recall": val_recall,
            "auc_score": auc_score / auc_count,
            "mcc": mcc_final,
            "bce_loss": val_loss_sum / loss_count,
        }

    def test(self, epoch):
        self.model.eval()
        device = self.args.device
        with torch.no_grad():
            test_no_train = list()
            total_name_list = list()
            total_length_list = list()

            for _, (
                set_max_len,
                data_name,
                data_length,
                contact,
                data_seq_raw,
                data_seq_encoding,
            ) in enumerate(self.test_loader):
                data_name = data_name.squeeze(0)
                contact = contact.squeeze(0)
                data_seq_encoding = data_seq_encoding.squeeze(0).to(device)
                data_length = data_length.squeeze(0)
                data_name_list = [
                    list(filter(lambda x: x != -1, item.numpy())) for item in data_name
                ]
                total_name_list += [decode_name(item) for item in data_name_list]
                total_length_list += [item for item in data_length]

                matrix_rep = torch.zeros_like(contact)
                data_length = data_length.to(device)
                contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                batch_size = contact.shape[0]
                pred_x0, _ = self.model.sample(
                    batch_size,
                    data_seq_raw,
                    set_max_len,
                    contact_masks,
                    data_seq_encoding,
                )

                pred_x0 = pred_x0.cpu().float()

                test_no_train_tmp = list(
                    map(
                        lambda i: rna_evaluation(
                            pred_x0[i].squeeze(), contact.float()[i].squeeze()
                        ),
                        range(pred_x0.shape[0]),
                    )
                )
                test_no_train += test_no_train_tmp

            accuracy, prec, recall, sens, spec, F1, MCC = zip(*test_no_train)

            f1_pre_rec_df = pd.DataFrame(
                {
                    "name": total_name_list,
                    "length": total_length_list,
                    "accuracy": list(np.array(accuracy)),
                    "precision": list(np.array(prec)),
                    "recall": list(np.array(recall)),
                    "sensitivity": list(np.array(sens)),
                    "specificity": list(np.array(spec)),
                    "f1": list(np.array(F1)),
                    "mcc": list(np.array(MCC)),
                }
            )

            accuracy = np.average(np.nan_to_num(np.array(accuracy)))
            precision = np.average(np.nan_to_num(np.array(prec)))
            recall = np.average(np.nan_to_num(np.array(recall)))
            sensitivity = np.average(np.nan_to_num(np.array(sens)))
            specificity = np.average(np.nan_to_num(np.array(spec)))
            F1 = np.average(np.nan_to_num(np.array(F1)))
            MCC = np.average(np.nan_to_num(np.array(MCC)))

            print("#" * 40)
            print("Average testing accuracy: ", round(accuracy, 3))
            print("Average testing F1 score: ", round(F1, 3))
            print("Average testing precision: ", round(precision, 3))
            print("Average testing recall: ", round(recall, 3))
            print("Average testing sensitivity: ", round(sensitivity, 3))
            print("Average testing specificity: ", round(specificity, 3))
            print("#" * 40)
            print("Average testing MCC", round(MCC, 3))
            print("#" * 40)
            print("")
        return {
            "f1": F1,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            "mcc": MCC,
        }, f1_pre_rec_df


class DataParallelDistribution(torch.nn.DataParallel):
    """
    A DataParallel wrapper for Distribution.
    To be used instead of nn.DataParallel for Distribution objects.
    """

    def log_prob(self, *args, **kwargs):
        return self.forward(*args, mode="log_prob", **kwargs)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)

    def sample_with_log_prob(self, *args, **kwargs):
        return self.module.sample_with_log_prob(*args, **kwargs)
