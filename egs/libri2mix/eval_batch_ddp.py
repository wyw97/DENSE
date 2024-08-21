# By Yiwen Wang, July 2024.

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

import torch.nn as nn 
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from asteroid.metrics import get_metrics
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates

from models.td_speakerbeam import TimeDomainSpeakerBeam
# from datasets.librimix_informed import LibriMixInformed
from datasets.librimix_predict import LibriMixPredict

import torch.multiprocessing as mp
import numpy as np 


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
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    compute_metrics = COMPUTE_METRICS
    if not conf["from_checkpoint"]:
        model = TimeDomainSpeakerBeam.from_pretrained(conf["model_path"])
    else:
        model = TimeDomainSpeakerBeam(**conf["train_conf"]["filterbank"],
                                   **conf["train_conf"]["masknet"],
                                   **conf["train_conf"]["enroll"]
                                   ) 
        ckpt = torch.load(conf["model_path"], map_location = torch.device('cpu')) 
        state_dict = {} 
        for k in ckpt['state_dict']: 
            state_dict[k.split('.',1)[1]] = ckpt['state_dict'][k]
        model.load_state_dict(state_dict)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()
    test_set = LibriMixPredict(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=conf["train_conf"]["data"]["segment"],
        segment_aux=conf["train_conf"]["data"]["segment_aux"],
        delay_sample=0,
        return_filename=True,
    )  # Uses all segment length

    sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    
    test_loader = DataLoader(
        test_set,
        # shuffle=False,
        batch_size=conf["train_conf"]["training"]["test_batch_size"],
        # num_workers=conf["train_conf"]["training"]["test_num_workers"],
        sampler=sampler,        
    )

    # local_save_dir = os.path.join(conf["out_dir"], "out_streaming_debug/")
    local_save_dir = conf["out_dir"]
    os.makedirs(local_save_dir, exist_ok=True)
    series_list = []
    torch.no_grad().__enter__()
    # Save some examples in a folder. Wav files and metrics as text.
    for idx, data in enumerate(tqdm(test_loader)):
        # if idx >= 1:
        #     break
        mix, source, enroll, _, mixture_path, source_path, enroll_path = data
        # print("mixture path: ", mixture_path, "source path: ", source_path, "enroll path: ", enroll_path)
        
        # break
        # mix, source, enroll = tensors_to_device([mix, source, enroll], device=model_device)
        mix = mix.cuda(rank)
        source = source.cuda(rank)
        enroll = enroll.cuda(rank)
        # mixture_name = test_set.mixture_path.split('/')[-1].split('.')[0]
        # print("shape: ", mix.shape, source.shape, enroll.shape)
        # torch.Size([6, 24000]) torch.Size([6, 1, 24000]) torch.Size([6, 24000])
        # est_source = model(mix, enroll)
        # est_source = model.forward_streaming(mix.unsqueeze(0), enroll.unsqueeze(0)).unsqueeze(0)
        est_source = model.module.forward_streaming(mix, enroll).unsqueeze(1)
        # print("est src shape: ", est_source.shape, mix.shape, source.shape, enroll.shape)
        # torch.Size([6, 1, 24000]) torch.Size([6, 24000]) torch.Size([6, 1, 24000]) torch.Size([6, 24000])
        # break
        mix_np = mix.cpu().data.numpy()
        source_np = source.cpu().data.numpy()
        est_source_np = est_source.squeeze(0).cpu().data.numpy()     
        # print("shape: ", mix_np.shape, source_np.shape, est_source_np.shape)
        # shape:  (6, 24000) (6, 1, 24000) (6, 1, 24000)
        # break   
        for file_idx in range(len(mixture_path)):
            mixture_name = source_path[file_idx].split("/")[-2] + "_" + source_path[file_idx].split("/")[-1].split(".")[0]
            sf.write(local_save_dir + f"/{mixture_name}_"
                                f"MIX.wav",
                mix[file_idx].cpu().data.numpy(), conf["sample_rate"])
            sf.write(local_save_dir + f"/{mixture_name}_"
                                f"SOURCE.wav",
                source[file_idx, 0].cpu().data.numpy(), conf["sample_rate"])

            # For each utterance, we get a dictionary with the mixture path,
            # the input and output metrics
            utt_metrics = get_metrics(
                mix_np[file_idx],
                source_np[file_idx],
                est_source_np[file_idx],
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
            )
            utt_metrics["mix_path"] = mixture_name
            est_source_np_normalized = normalize_estimates(est_source_np[file_idx], mix_np[file_idx])
            series_list.append(pd.Series(utt_metrics))
            
            sf.write(local_save_dir + f"/{mixture_name}_"
                                    f"s.wav",
                    est_source_np_normalized[0], conf["sample_rate"])
        break
    dist.destroy_process_group()
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

    print("Overall metrics :")
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
        
    # random_seed = 2203 
    # torch.manual_seed(random_seed)
    # random.seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) 
    # np.random.seed(random_seed)
    # main(arg_dic) 
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, arg_dic), nprocs=world_size, join=True)
