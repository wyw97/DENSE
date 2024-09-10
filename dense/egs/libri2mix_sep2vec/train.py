# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

# This code is based on Asteroid train.py, 
# which is released under the following MIT license:
# https://github.com/asteroid-team/asteroid/blob/master/LICENSE

import os
import argparse
import json

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from collections import OrderedDict

from models_direct.td_speakerbeam_sep2vec import TimeDomainSpeakerBeamPredictHelp
# from models.td_speakerbeam_skim import TimeDomainSpeakerBeamPredictHelp
from datasets.librimix_predict import LibriMixPredict
from asteroid.engine.optimizers import make_optimizer
# from models_direct.system import SystemPredicted, SystemPredictedTeacherForcing, SystemPredictedTeacherForcingAutoRegression, SystemPredictedTeacherForcingTripleTime
from models_direct.system import SystemPredictedPairs
from asteroid.losses import singlesrc_neg_sisdr, singlesrc_neg_snr
import tensorboard

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

def loss(pred, tgt):
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

def neg_sisdr_loss_wrapper(est_targets, targets):
    return singlesrc_neg_sisdr(est_targets[:,0], targets[:,0]).mean()

def neg_snr_loss_wrapper(est_targets, targets):
    return singlesrc_neg_snr(est_targets[:,0], targets[:,0]).mean()

def main(conf):
    train_set = LibriMixPredict(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
        delay_sample=conf["data"]["delay_sample"],
    )

    val_set = LibriMixPredict(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
        delay_sample=conf["data"]["delay_sample"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    model = TimeDomainSpeakerBeamPredictHelp(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"],
        **conf["enroll"]
    )
    # model = TimeDomainSpeakerBeamPredictHelp(i_adapt_layer=7, adapt_layer_type='mul', adapt_enroll_dim=128, causal=True)
    # load from pretrained model
    model_path = "/data1/wangyiwen/repos/graduateproject/speakerbeam/egs/libri2mix/exp/20240629_mixloss/best_model.pth"
    # model_path = None
    if model_path:
        model_state_dict = torch.load(model_path)['state_dict']
        print("state_dict keys: ", model_state_dict.keys())
        # model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items() if "model." in k}
        new_state_dict = OrderedDict()
        for name, param in model.named_parameters():
            if name in model_state_dict:
                new_state_dict[name] = model_state_dict[name]
                print(name)
            else:
                # print("not found: ", name)
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict)
        print("load successfully...")
        for name, param in model.named_parameters():
            if name in model_state_dict:
                # pass
                param.requires_grad = False
    # return 
    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=conf['training']['reduce_patience'])
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    # loss_func = neg_sisdr_loss_wrapper
    # loss_func = nn.L1Loss()
    # loss_func = loss
    loss_func = neg_snr_loss_wrapper
    # system = SystemPredictedTeacherForcingAutoRegression(
    system = SystemPredictedPairs(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=-1, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        # callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=conf['training']['stop_patience'], verbose=True))
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=conf["training"]["epochs"], verbose=True))
    callbacks.append(LearningRateMonitor())

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        devices="auto",
        # distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    # trainer.fit(system, ckpt_path="/data1/wangyiwen/repos/graduateproject/speakerbeam/egs/libri2mix_sep2vec/exp/auto_regression_0_3_16sample_cgLN/checkpoints/epoch=166-step=96693.ckpt")
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)
    # print("best model path: ", checkpoint.best_model_path)
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
