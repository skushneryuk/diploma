# Вспомогательные
import gc
import os
from collections import defaultdict
from packaging import version

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

# Библиотеки для обучения
import torch
import pytorch_lightning as pl

# Библиотеки для обработки изображений
# Библиотеки для визуализации
from IPython.display import clear_output
from IPython import display


from .unet import UNet
from .losses import *
from metrics import *
from datasets import *


class LightningSegmentation(pl.LightningModule):
    def __init__(self, model, loss_func=dice_loss, loss_params=dict(), dataloaders=dict(),
                 optim_cls=torch.optim.Adam, optim_params=dict(),
                 scheduler_cls=None, scheduler_params=dict(),
                 visualize_step=100, checkpointing=False, checkpoint_path=None,
                 compile=False, checkpoint_metric_name="valid_dice",
                 valid_eps_restart=0.01, valid_n_restart=5,):
        super().__init__()

        self.model = model
        self.dataloaders = dataloaders
        self.checkpointing = checkpointing
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metric_name = checkpoint_metric_name
        self.compile = compile
        if compile:
            model = torch.compile(model)
            print('\t\t==============================================')
            print('\t\t============= Model compiled =================')
            print('\t\t==============================================')

        self.valid_eps_restart = valid_eps_restart
        self.valid_n_restart = valid_n_restart
        self.restart_flag = False

        self.optim_params = optim_params
        self.scheduler_params = scheduler_params
        self.optimizer = optim_cls(self.model.parameters(), **self.optim_params)
        self.scheduler = None
        if scheduler_cls is not None:
            self.scheduler = scheduler_cls(self.optimizer, **self.scheduler_params)

        self.loss_func = loss_func
        self.loss_params = loss_params

        assert (not checkpointing) or (checkpoint_path is not None)
        assert loss_func is not None

        self.info = defaultdict(list)
        self.visualize_step = visualize_step

        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
        self.dh = display.display(self.fig, display_id=True)

    def clear_cache(self, k=3):
        for k in range(k):
            torch.cuda.empty_cache()
            gc.collect()

    def save_model(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

        if self.compile:
            torch.save(self.model._orig_mod.cpu().state_dict(), self.checkpoint_path)
        else:
            torch.save(self.model.cpu().state_dict(), self.checkpoint_path)
        self.model = self.model.cuda()
        print("\t\t======================= MODEL SAVED ==========================")

    def load_model(self, path=None):
        if path is None:
            path = self.checkpoint_path

        if self.compile and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod

        self.model.load_state_dict(torch.load(path))
        self.model = self.model.cuda()

        if self.compile:
            self.model = torch.compile(self.model).cuda()
        print("\t\t======================= MODEL LOADED =========================")

    def check_metric_and_checkpoint(self):
        if len(self.info[self.checkpoint_metric_name]) <= 1:
            self.save_model()
            return

        best_metric = max(self.info[self.checkpoint_metric_name])
        if self.info[self.checkpoint_metric_name][-1] == best_metric:
            self.save_model()

    def configure_optimizers(self):
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
            }
        return self.optimizer

    def compute_loss(self, logits, labels):
        return self.loss_func(labels, logits, **self.loss_params)

    def restart_model_if_needed(self):
        if self.restart_flag:
            self.load_model()
            self.optimizer.__setstate__({'state': defaultdict(dict)})
            self.restart_flag = False

    def training_step(self, batch, batch_idx):
        self.restart_model_if_needed()
        image = batch['image']
        labels = batch['mask']

        logits = self.model(image)
        loss = self.compute_loss(logits, labels)

        self.info['train_loss'].append(loss.detach().cpu().numpy().mean())
        self.log("train_loss", self.info['train_loss'][-1])

        if batch_idx % self.visualize_step == 0:
            self.ax[0].clear()
            self.ax[0].plot(self.info['train_loss'])
            self.dh.update(self.fig)

        if torch.all(torch.isnan(loss)).item():
            self.restart_flag = True

        return loss

    def on_train_epoch_end(self):
        self.ax[0].clear()
        self.ax[0].plot(self.info['train_loss'])
        self.dh.update(self.fig)

        self.clear_cache()
        self.restart_model_if_needed()

    def validation_step(self, batch, _):
        image = batch['image']
        labels = batch['mask']

        with torch.no_grad():
            preds = self.model.predict(image)
            self.info['valid_preds'].append(preds.detach().cpu().numpy().squeeze())
            self.info['valid_labels'].append(labels.detach().cpu().numpy().squeeze())

    def on_validation_epoch_end(self):
        valid_preds = np.concatenate(self.info['valid_preds'])
        valid_labels = np.concatenate(self.info['valid_labels'])

        self.info['valid_preds'] = []
        self.info['valid_labels'] = []

        valid_dice = dice_metric(valid_labels, valid_preds, reduce=False)
        valid_iou = iou_metric(valid_labels, valid_preds, reduce=False)
        valid_recall = recall_metric(valid_labels, valid_preds, reduce=False)
        valid_precision = precision_metric(valid_labels, valid_preds, reduce=False)

        valid_dice_mean = valid_dice.mean()
        valid_iou_mean = valid_iou.mean()
        valid_recall_mean = valid_recall.mean()
        valid_precision_mean = valid_precision.mean()

        self.info['valid_dice'].append(valid_dice_mean)
        self.info['valid_iou'].append(valid_iou_mean)
        self.info['valid_recall'].append(valid_recall_mean)
        self.info['valid_precision'].append(valid_precision_mean)

        self.log("valid_dice", valid_dice_mean)

        self.ax[1].clear()
        self.ax[1].plot(self.info['valid_dice'], label='DICE')
        self.ax[1].plot(self.info['valid_iou'], label='IOU')
        self.ax[1].plot(self.info['valid_recall'], label='RECALL')
        self.ax[1].plot(self.info['valid_precision'], label='PRECISION')
        self.ax[1].legend()
        self.dh.update(self.fig)

        self.clear_cache()
        self.check_metric_and_checkpoint()

        if len(self.info['valid_recall']) >= self.valid_n_restart and \
           np.all(np.array(self.info['valid_recall'][-self.valid_n_restart:]) <= self.valid_eps_restart) and \
           np.all(np.array(self.info['valid_precision'][-self.valid_n_restart:]) >= 1. - self.valid_eps_restart):
            print("\t\t======================= OVERFITTING ==========================")
            self.restart_flag = True

        self.restart_model_if_needed()

    def test_step(self, batch, _):
        image = batch['image']
        labels = batch['mask']

        with torch.no_grad():
            preds = self.model.predict(image)
            self.info['test_preds'].append(preds.detach().cpu().numpy().squeeze())
            self.info['test_labels'].append(labels.detach().cpu().numpy().squeeze())

    def on_test_epoch_end(self):
        test_preds = np.concatenate(self.info['test_preds'])
        test_labels = np.concatenate(self.info['test_labels'])

        self.info['test_preds'] = []
        self.info['test_labels'] = []

        test_dice = dice_metric(test_labels, test_preds, reduce=False)
        test_iou = iou_metric(test_labels, test_preds, reduce=False)
        test_recall = recall_metric(test_labels, test_preds, reduce=False)
        test_precision = precision_metric(test_labels, test_preds, reduce=False)

        test_dice_mean = test_dice.mean()
        test_iou_mean = test_iou.mean()
        test_recall_mean = test_recall.mean()
        test_precision_mean = test_precision.mean()

        test_dice_var = test_dice.var(ddof=1)
        test_iou_var = test_iou.var(ddof=1)
        test_recall_var = test_recall.var(ddof=1)
        test_precision_var = test_precision.var(ddof=1)

        test_dice_q025 = np.quantile(test_dice, 0.025)
        test_iou_q025 = np.quantile(test_iou, 0.025)
        test_recall_q025 = np.quantile(test_recall, 0.025)
        test_precision_q025 = np.quantile(test_precision, 0.025)

        test_dice_q975 = np.quantile(test_dice, 0.975)
        test_iou_q975 = np.quantile(test_iou, 0.975)
        test_recall_q975 = np.quantile(test_recall, 0.975)
        test_precision_q975 = np.quantile(test_precision, 0.975)

        print()
        print("\t\t==============================================================")
        print("\tTEST DICE: mean = {:.4f}, std = {:.4f}, q2.5% = {:.4f}, q97.5% = {:.4f}".\
              format(test_dice_mean, test_dice_var, test_dice_q025, test_dice_q975))
        print("\tTEST IOU: mean = {:.4f}, std = {:.4f}, q2.5% = {:.4f}, q97.5% = {:.4f}".\
              format(test_iou_mean, test_iou_var, test_iou_q025, test_iou_q975))
        print("\tTEST RECALL: mean = {:.4f}, std = {:.4f}, q2.5% = {:.4f}, q97.5% = {:.4f}".\
              format(test_recall_mean, test_recall_var, test_recall_q025, test_recall_q975))
        print("\tTEST PRECISION: mean = {:.4f}, std = {:.4f}, q2.5% = {:.4f}, q97.5% = {:.4f}".\
              format(test_precision_mean, test_precision_var, test_precision_q025, test_precision_q975))
        print("\t\t==============================================================")
        print()

        self.log("test_dice_mean", test_dice_mean)
        self.log("test_dice_var", test_dice_var)
        self.log("test_dice_q025", test_dice_q025)
        self.log("test_dice_q975", test_dice_q975)

        self.log("test_iou_mean", test_iou_mean)
        self.log("test_iou_var", test_iou_var)
        self.log("test_iou_q025", test_iou_q025)
        self.log("test_iou_q975", test_iou_q975)

        self.log("test_recall_mean", test_recall_mean)
        self.log("test_recall_var", test_recall_var)
        self.log("test_recall_q025", test_recall_q025)
        self.log("test_recall_q975", test_recall_q975)

        self.log("test_precision_mean", test_precision_mean)
        self.log("test_precision_var", test_precision_var)
        self.log("test_precision_q025", test_precision_q025)
        self.log("test_precision_q975", test_precision_q975)

        self.clear_cache()

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['valid']

    def test_dataloader(self):
        return self.dataloaders['test']


def setup_cityscapes_experiment(model_params, loss_func, loss_params, dataset_params,
                     optim_params, trainer_params, early_stopping_params,
                     experiment_name, visualize_step=100, weight_save_path=None,
                     checkpoint_metric_name="valid_dice", compile=False,
                     scheduler_cls=None, scheduler_params=dict(), restart=False,
                     valid_n_restart=10):
    for k in range(5):
        torch.cuda.empty_cache()
        gc.collect()

    print('\t\t==============================================')
    print(f'\t\t {experiment_name} ')
    print('\t\t==============================================')

    compile = compile and version.parse(torch.__version__) >= version.parse("2.0.0")
    checkpointing = weight_save_path is not None
    checkpoint_path = None
    if checkpointing:
        checkpoint_path = os.path.join(weight_save_path, experiment_name)

    plModule = LightningSegmentation(
        UNet(**model_params),
        loss_func=loss_func,
        loss_params=loss_params,
        dataloaders=setup_cityscapes_dataloaders(**dataset_params),
        optim_params=optim_params,
        visualize_step=visualize_step,
        checkpointing=checkpointing,
        checkpoint_path=checkpoint_path,
        checkpoint_metric_name=checkpoint_metric_name,
        scheduler_cls=scheduler_cls,
        scheduler_params=scheduler_params,
        valid_n_restart=valid_n_restart,
        compile=compile,
    )
    if restart:
        plModule.load_model()

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(**early_stopping_params)

    trainer = pl.Trainer(
        **trainer_params,
        callbacks=[early_stopping],
    )

    print("\t\t==================== STARTING TRAINING =======================")
    trainer.fit(plModule)
    print("\t\t==================== TRAINING ENDED    =======================")
    plModule.load_model()
    print("\t\t==================== STARTING TESTING  =======================")
    trainer.test(plModule)
    print("\t\t==================== TESTING ENDED     =======================")


def setup_cococars_experiment(model_params, loss_func, loss_params, dataset_params,
                     optim_params, trainer_params, early_stopping_params,
                     experiment_name, visualize_step=100, weight_save_path=None,
                     checkpoint_metric_name="valid_dice", compile=False,
                     scheduler_cls=None, scheduler_params=dict(), restart=False,
                     valid_n_restart=10):
    for k in range(5):
        torch.cuda.empty_cache()
        gc.collect()

    print('\t\t==============================================')
    print(f'\t\t {experiment_name} ')
    print('\t\t==============================================')

    compile = compile and version.parse(torch.__version__) >= version.parse("2.0.0")
    checkpointing = weight_save_path is not None
    checkpoint_path = None
    if checkpointing:
        checkpoint_path = os.path.join(weight_save_path, experiment_name)

    plModule = LightningSegmentation(
        UNet(**model_params),
        loss_func=loss_func,
        loss_params=loss_params,
        dataloaders=setup_cococars_dataloaders(**dataset_params),
        optim_params=optim_params,
        visualize_step=visualize_step,
        checkpointing=checkpointing,
        checkpoint_path=checkpoint_path,
        checkpoint_metric_name=checkpoint_metric_name,
        scheduler_cls=scheduler_cls,
        scheduler_params=scheduler_params,
        valid_n_restart=valid_n_restart,
        compile=compile,
    )
    if restart:
        plModule.load_model()

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(**early_stopping_params)

    trainer = pl.Trainer(
        **trainer_params,
        callbacks=[early_stopping],
    )

    print("\t\t==================== STARTING TRAINING =======================")
    trainer.fit(plModule)
    print("\t\t==================== TRAINING ENDED    =======================")
    plModule.load_model()
    print("\t\t==================== STARTING TESTING  =======================")
    trainer.test(plModule)
    print("\t\t==================== TESTING ENDED     =======================")


def setup_cocostuff_experiment(model_params, loss_func, loss_params, dataset_params,
                     optim_params, trainer_params, early_stopping_params,
                     experiment_name, visualize_step=100, weight_save_path=None,
                     checkpoint_metric_name="valid_dice", compile=False,
                     scheduler_cls=None, scheduler_params=dict(), restart=False):
    for k in range(5):
        torch.cuda.empty_cache()
        gc.collect()

    print('\t\t==============================================')
    print(f'\t\t {experiment_name} ')
    print('\t\t==============================================')

    compile = compile and version.parse(torch.__version__) >= version.parse("2.0.0")
    checkpointing = weight_save_path is not None
    checkpoint_path = None
    if checkpointing:
        checkpoint_path = os.path.join(weight_save_path, experiment_name)

    plModule = LightningSegmentation(
        UNet(**model_params),
        loss_func=loss_func,
        loss_params=loss_params,
        dataloaders=setup_cocostuff_dataloaders(**dataset_params),
        optim_params=optim_params,
        visualize_step=visualize_step,
        checkpointing=checkpointing,
        checkpoint_path=checkpoint_path,
        checkpoint_metric_name=checkpoint_metric_name,
        scheduler_cls=scheduler_cls,
        scheduler_params=scheduler_params,
        compile=compile,
    )
    if restart:
        plModule.load_model()

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(**early_stopping_params)

    trainer = pl.Trainer(
        **trainer_params,
        callbacks=[early_stopping],
    )

    print("\t\t==================== STARTING TRAINING =======================")
    trainer.fit(plModule)
    print("\t\t==================== TRAINING ENDED    =======================")
    plModule.load_model()
    print("\t\t==================== STARTING TESTING  =======================")
    trainer.test(plModule)
    print("\t\t==================== TESTING ENDED     =======================")


def get_general_params(base_lr=1e-3, scheduler_cls=None, scheduler_params=dict(),
                       patience=10, compile=False, max_epoch=100, visualize_step=100):
    general_params = {
        "model_params": {
            "n_channels": 3,
            "n_classes": 2,
            "depth": 3,
            "start_channels": 32,
            "upsample": False,
        },
        "dataset_params": dict(),
        "optim_params": {
            "lr": base_lr,
            "eps": 1e-4,
        },
        "trainer_params": {
            "max_epochs": max_epoch,
            "accumulate_grad_batches": 1,
            "accelerator": 'gpu',
            "gpus": 1,
            "val_check_interval": 0.5,
            "checkpoint_callback": False,
        },
        "early_stopping_params": {
            "monitor": "valid_dice",
            "min_delta": 0.0,
            "patience": patience,
            "mode":"max",
        },
        "weight_save_path": "./checkpoints/",
        "visualize_step": visualize_step,
        "checkpoint_metric_name": "valid_dice",
        "compile": compile,
    }
    if scheduler_cls is not None:
        general_params['scheduler_cls'] = scheduler_cls
        general_params['scheduler_params'] = scheduler_params

    return general_params