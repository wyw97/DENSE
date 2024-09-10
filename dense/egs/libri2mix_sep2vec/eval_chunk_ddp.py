# All rights reserved
# Yiwen Wang, August 2024.

# This code is based on Asteroid eval.py, 
# which is released under the following MIT license:
# https://github.com/asteroid-team/asteroid/blob/master/LICENSE

import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

import numpy as np 

import torch.nn as nn 
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from asteroid.metrics import get_metrics
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates

from models_direct.td_speakerbeam_sep2vec import TimeDomainSpeakerBeamPredictHelp 
from datasets.librimix_predict import LibriMixPredict, LibriMixPredictWithSep

import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the model (either best model or checkpoint, "
         "which is determined by parameter --from_checkpoint"
)

parser.add_argument(
    "--from_checkpoint",
    type=int,
    default=0,
    help="Model in model path is checkpoint, not final model. Default: 0"
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def set_seed(random_seed=2203):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

def main(rank, world_size, conf):
    set_seed(2203)
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365' # 12345
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    compute_metrics = COMPUTE_METRICS
    if not conf["from_checkpoint"]:
        model = TimeDomainSpeakerBeamPredictHelp.from_pretrained(conf["model_path"])
    else:
        model = TimeDomainSpeakerBeamPredictHelp(
                **conf['train_conf']["filterbank"], **conf['train_conf']["masknet"], sample_rate=conf['train_conf']["data"]["sample_rate"],
                **conf['train_conf']["enroll"]
        )
        ckpt = torch.load(conf["model_path"], map_location = torch.device('cpu')) 
        state_dict = {} 
        for k in ckpt['state_dict']: 
            state_dict[k.split('.',1)[1]] = ckpt['state_dict'][k]
        # print(model.state_dict().keys())
        model.load_state_dict(state_dict)
    
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # model_device = next(model.parameters()).device
    model.eval()
    # test with actual answer
    test_set = LibriMixPredict(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=conf["train_conf"]["data"]["segment"],
        segment_aux=conf["train_conf"]["data"]["segment_aux"],
        delay_sample=conf["train_conf"]["data"]["delay_sample"],
        return_filename=True,
    )  # Uses all segment length
    sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(
        test_set,
        batch_size=conf["train_conf"]["training"]["test_batch_size"],
        sampler=sampler,
    )
    
    series_list = []
    local_save_dir = conf["out_dir"]
    os.makedirs(local_save_dir, exist_ok=True)
    torch.no_grad().__enter__()
    for idx, data in enumerate(tqdm(test_loader)):
        # Forward the network on the mixture.
        mix, source, enroll, delay_src, mixture_path, source_path, enroll_path = data
        mix = mix.cuda(rank)
        source = source.cuda(rank)
        enroll = enroll.cuda(rank)
        delay_src = delay_src.cuda(rank)
        est_source = model.module.forward_streaming(mix, enroll, kernel_size=16, step_size=8, time_delay=conf["train_conf"]["data"]["delay_sample"]).unsqueeze(1)
        mix_np = mix.cpu().data.numpy()
        source_np = source.cpu().data.numpy()
        est_source_np = est_source.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        for file_idx in range(len(mixture_path)):
            mixture_name = source_path[file_idx].split("/")[-2] + "_" + source_path[file_idx].split("/")[-1].split(".")[0]
            sf.write(local_save_dir + f"/{mixture_name}_"
                                f"MIX.wav",
                mix[file_idx].cpu().data.numpy(), conf["sample_rate"])
            sf.write(local_save_dir + f"/{mixture_name}_"
                                f"SOURCE.wav",
                source[file_idx, 0].cpu().data.numpy(), conf["sample_rate"])
            utt_metrics = get_metrics(
                mix_np[file_idx],
                source_np[file_idx],
                est_source_np[file_idx],
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
            )
            utt_metrics["mix_path"] = mixture_name
            est_source_np_normalized = normalize_estimates(est_source_np[file_idx], mix_np[file_idx])
            # print("shape : ", est_source_np_normalized.shape)
            series_list.append(pd.Series(utt_metrics))
            sf.write(local_save_dir + f"/{mixture_name}_"
                                    f"s.wav",
                    est_source_np_normalized[0], conf["sample_rate"])
        # break 
    # return 
    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(local_save_dir, f"all_metrics_rank{rank}.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print(f"Overall metrics {rank}:")
    pprint(final_results)

    with open(os.path.join(local_save_dir, f"final_metrics_rank{rank}.json"), "w") as f:
        json.dump(final_results, f, indent=0)



if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )
    print(arg_dic)

    world_size =torch.cuda.device_count()
    mp.spawn(main, args=(world_size, arg_dic), nprocs=world_size, join=True)
