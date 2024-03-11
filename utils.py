import os
import typing
from scipy.ndimage import zoom
import os, shutil
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import time
import torch
from torch.utils.data import DataLoader 
import numpy as np
from PIL import Image
import wandb
import torch
from einops import rearrange
import cv2
import k3d
import cv2 
import numpy as np
import torch
import tqdm
import skimage.draw
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr


def get_reference_videos(config, val_reference, indices, stage, scale=255, clamp_range=(0,255)):
    vformat = lambda x: (x*scale).permute(1,2,3,0).numpy().astype(np.uint8)
    videos = [ vformat(val_reference[i]) for i in range(len(indices)) ]
    # Resize depending on unet
    undex_index = stage - 1 # always input size which is the last 
    size = config.imagen.image_sizes[undex_index]
    temp_down = config.imagen.temporal_downsample_factor[undex_index]
    videos = [ zoom(v, (1/temp_down, size/v.shape[1], size/v.shape[2], 1)).clip(clamp_range) for v in videos ]

    return videos

def delay2str(t):
    t = int(t)
    secs = t%60
    mins = (t//60)%60
    hours = (t//3600)%24
    days = t//86400
    string = f"{secs}s"
    if mins:
        string = f"{mins}m {string}"
    if hours:
        string = f"{hours}h {string}"
    if days:
        string = f"{days}d {string}"
    return string


def create_save_folder(args ,save_folder):
    os.makedirs(save_folder, exist_ok = True)
    shutil.copy(args.config, os.path.join(save_folder, "config.yaml"))
    os.makedirs(os.path.join(save_folder, "videos"), exist_ok = True)
    os.makedirs(os.path.join(save_folder, "models"), exist_ok = True)

def delete_save_folder(save_folder):
    try:
        shutil.rmtree(save_folder)
    except:
        pass

def visualize_volume(volume, title:str):
    if type(volume) is not np.ndarray:
        if torch.is_tensor(volume):
            volume = volume.numpy()

    plot = k3d.plot()
    volume = k3d.volume(volume.astype(np.float32), color_map=k3d.basic_color_maps.Jet, scaling=[1,1,1])
    plot += volume
    plot.display()

    with open('k3d.html', 'w') as f:
        f.write(plot.get_snapshot())
    wandb.log({title: wandb.Html(open("k3d.html"), inject=False)})

def one_line_log(config, cur_step, loss, batch_per_epoch, start_time, validation=False):
    s_step = f'Step: {cur_step:<6}'
    s_loss = f'Loss: {loss:<6.4f}' if not validation else f'Val loss: {loss:<6.4f}'
    s_epoch = f'Epoch: {(cur_step//batch_per_epoch):<4.0f}'
    s_mvid = f'Mvid: {(cur_step*config.dataloader.batch_size/1e6):<6.4f}'
    s_delay = f'Elapsed time: {delay2str(time.time() - start_time):<10}'
    print(f'{s_step} | {s_loss} {s_epoch} {s_mvid} | {s_delay}', end='\r') # type: ignore
    if cur_step % 1000 == 0:
        print() # Start new line every 1000 steps
    
    wandb.log({
        "loss" if not validation else "val_loss" : loss, 
        "step": cur_step, 
        "epoch": cur_step//batch_per_epoch, 
        "mvid": cur_step*config.dataloader.batch_size/1e6
    })

def start_wandb(config, exp_name, train_days):
    wandb.init(
        name=f"{exp_name}_[{train_days}d]",
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True) # type: ignore
    )

def timestamp():
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{t}]:"

def vid_mean_ssim(video_pred, video_tar):
    frame_ssims = []
    for (frame_pred, frame_tar) in list(zip(video_pred, video_tar)):
        frame_ssim = ssim(frame_pred, frame_tar, data_range =1)
        frame_ssims.append(frame_ssim)
    return sum(frame_ssims) / len(frame_ssims)

def vid_mean_mse(video_pred, video_tar):
    frame_mses = []
    for (frame_pred, frame_tar) in list(zip(video_pred, video_tar)):
        frame_mse = mean_squared_error(frame_pred, frame_tar)
        frame_mses.append(frame_mse)
    return sum(frame_mses) / len(frame_mses)

def vid_mean_psnr(video_pred, video_tar):
    frame_psnrs = []
    for (frame_pred, frame_tar) in list(zip(video_pred, video_tar)):
        frame_psnr = psnr(frame_pred, frame_tar, data_range=1)
        frame_psnrs.append(frame_psnr)
    return sum(frame_psnrs) / len(frame_psnrs)


