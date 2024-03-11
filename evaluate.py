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
from scipy.ndimage import zoom
import cv2
import pandas as pd 
import k3d

from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
# from diffusion.dataset import EchoVideo, UKBBPartial
from diffusion.dataset import EchoVideo, UKBBPartial
from utils import get_reference_videos, vid_mean_mse, vid_mean_ssim, vid_mean_psnr

### !!! Allowed to call validation dataset only one time, otherwise we always get random slices !!! 


if __name__ == "__main__":
    # Get config and args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to merged folder")
    parser.add_argument("--bs", type=int, default="1")
    parser.add_argument("--cond_scale", type=float, default=5., help="Evaluation batch size.")
    parser.add_argument("--chunks", type=int, default=1, help="Number of diffusion steps.")
    parser.add_argument("--chunk", type=int, default=0, help="Number of diffusion steps.")
    parser.add_argument("--stop_at", type=float, default=-1, help="stop a certain UNET")
    parser.add_argument("--ef_list", type=str, help="path to csv containing all requested EFs")

    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.model, "config.yaml"))
    config = OmegaConf.merge(config, vars(args))

    print(f"Started work on chunk {args.chunk} of {args.chunks} at {datetime.now()}")

    # Overwrite config values with args ____???????? ICL specific
    config.dataset.num_frames = int(config.dataset.fps * config.dataset.duration)
    if os.uname().nodename.endswith("doc.ic.ac.uk"):
        config.dataset.data_path = "/data/hjr119/echo_ds/EchoNet-Dynamic"
        config.checkpoint.path = "/data/hjr119/diffusion_models/imagen_video"
    if args.stop_at == -1:
        args.stop_at = None

    # Get current exp name
    exp_name = args.model.split("/")[-1]

    # Load dataset
    print(config.bs)
    dataset = UKBBPartial(config)
    dataloader = DataLoader(dataset=dataset, batch_size=config.bs)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    unets = []
    for i, (k, v) in enumerate(config.unets.items()):
        unets.append(Unet3D(**v, lowres_cond=(i>0))) # type: ignore

    del config.imagen.elucidated

    imagen = ElucidatedImagen(
        unets = unets,
        **OmegaConf.to_container(config.imagen), # type: ignore
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        **config.trainer
    ).to(device)

    # Load model
    path_to_model = os.path.join(args.model, "merged.pt")
    assert os.path.exists(path_to_model), f"Model {path_to_model} does not exist. Did you merge the checkpoints?"
    additional_data = trainer.load(path_to_model)
    print(f"Loaded model {path_to_model}")
    trainer.eval()



    save_folder = '/home/niklas/Desktop'
    print(save_folder)


    df = pd.DataFrame(columns=['fname', 'mse1', 'mse2', 'mse3', 'mse4', 'ssim1', 'ssim2', 'ssim3', 'ssim4', 'psnr1', 'psnr2', 'psnr3', 'psnr4'])

    # validate dataset 
    for data in dataloader:
        print(data[3])
        # Get all the validation data / only call once because we have random in dataset.__getItem__
        val_reference = torch.stack([data[0][i] for i in range(config.bs)])
        val_cond_images = torch.stack([data[1][i] for i in range(config.bs)])
        val_embeddings = torch.stack([data[2][i] for i in range(config.bs)])
        val_fnames = [data[3][i] for i in range(config.bs)]

        ref_list = []
        for i, (k, v) in enumerate(config.unets.items()):
            stage = i + 1
            videos_ref = get_reference_videos(config, val_reference, range(config.bs), stage=stage, scale=1)
            videos_ref = np.concatenate([np.array(img) for img in videos_ref], axis=-2) # T x H x (W B) x C
            videos_ref = rearrange(videos_ref, 't h (w b) c -> b c t h w', b=config.bs)
            ref_list.append(videos_ref)

            print(f"Created Reference video: Stage{stage} {videos_ref.shape}")
            
            # # save videos 
            # _T, _H, _W, _C = videos_ref.shape
            # videos_pil = [Image.fromarray(i[...,0]) for i in videos_ref]
            # if trainer.is_main:
            #     videos_pil[0].save(os.path.join(save_folder, "videos", f"_reference.gif"), save_all=True, append_images=videos_pil[1:], duration=1000/config.dataset.fps, loop=0)
            #     print("Saved reference videos.")


        print("emebds: ", val_embeddings)
        valid_embeddings = val_embeddings
        # valid_embeddings[0,0,0] = 14
        # print("emebds: ", valid_embeddings)
        # Create kwargs for sampling and logging

        sample_kwargs = {}
        sample_kwargs["start_at_unet_number"] = 1
        sample_kwargs["stop_at_unet_number"] = 2
        if config.imagen.condition_on_text:
            sample_kwargs["text_embeds"] = valid_embeddings
        # if config.unets.get(f"unet{args.stage}").get('cond_images_channels') > 0:
        sample_kwargs["cond_images"] = val_cond_images
        
        # sample_kwargs["start_image_or_video"] = val_reference # type: ignore

        sample_kwargs["return_all_unet_outputs"] = True

        s_videos = trainer.sample(
                            batch_size=config.checkpoint.batch_size, 
                            cond_scale=config.checkpoint.cond_scale,
                            video_frames=config.dataset.num_frames,
                            **sample_kwargs,
                        ) # B x C x T x H x W
        

        
        for b in range(config.bs):
            row = {}
            for i in range(len(s_videos)):
    
                ssim = vid_mean_ssim(s_videos[i][b, 0, ...].cpu().numpy(), ref_list[i][b, 0, ...])
                mse = vid_mean_mse(s_videos[i][b, 0, ...].cpu().numpy(), ref_list[i][b, 0, ...])
                psnr = vid_mean_psnr(s_videos[i][b, 0, ...].cpu().numpy(), ref_list[i][b, 0, ...])
                
                row['fname'] = val_fnames[b]
                row[f'mse{i+1}'] = mse
                row[f'ssim{i+1}'] = ssim
                row[f'psnr{i+1}'] = psnr
            df.loc[len(df.index)] = row

        print(df.head(5))

        videos = rearrange(s_videos.detach().cpu(), 'b c t h w -> b t c h w') # type: ignore
        videos = (videos*255).numpy().astype(np.uint8)
        
        print(videos.shape, videos_ref.shape)
        videos = rearrange(videos, 'b t c h w -> t c h (b w)') # type: ignore

        videos = rearrange(s_videos.detach().cpu(), 'b c t h w -> t h (b w) c') # type: ignore
        videos = (videos*255).numpy().astype(np.uint8)
        print(videos.shape)
        videos = [Image.fromarray(i[...,0]) for i in videos]
        videos[0].save(os.path.join(save_folder, "videos", f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/config.dataset.fps, loop=0) 

