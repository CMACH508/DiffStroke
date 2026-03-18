import argparse
import logging
import os
import os.path as osp
import time
from einops import rearrange, repeat
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
from PIL import Image
import torch.nn.functional as F
from dist_util import get_bare_model, get_dist_info, init_dist, master_only
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.model_edge import pidinet
from ldm.modules.fusion_block import fusionnet
from Dataset import Sketchy_data, Custom_dataset
import random
os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='2075'
seed = 1010
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
np.random.seed(seed)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=8,
    help="the prompt to render"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10000,
    help="the prompt to render"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="the prompt to render"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="the prompt to render"
)
parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
)
parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--data_path",
        type=str,
        #default='./Dataset/Generate_data/Places2',
        #default='./Dataset/Generate_data/CelebA-HQ',
        #default='./Dataset/Generate_data/Users',
        default='./Dataset/Generate_data/Nature_CVPR',
        #default='./Dataset/Generate_data/Custom_Face',
        #default='./Dataset/Generate_data/Nature_CVPR',
        #default='./Dataset/Generate_data/Face_CVPR',
        #default='./Dataset/Generate_data/Face_CVPR_div',
        #default='./Dataset/Generate_data/Nature_CVPR_div',
        #default='./Dataset/Generate_data/Nature2',
        #default='./Dataset/Generate_data/Face2',
        #default='./Dataset/Generate_data/Failure',
        help="path of dataset",
)
parser.add_argument(
        "--model_path",
        type=str,
        # default= '/home/litengjie/MaskSketch/experiments/train_sketch_weight_1_1_1/models/model_fusion_170000.pth',
        #default= '/home/litengjie/MaskSketch/experiments/train_sketch_all_mask/models/model_fusion_170000.pth',
        default = './experiments/MaskV2/models/model_fusion_170000.pth',
        #default='./experiments/face_finetune/models/model_fusion_30000.pth',
        help="path of model",
)
parser.add_argument(
        "--ckpt",
        type=str,
        default="models/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
)
parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/train_sketch.yaml",
        help="path to config which constructs model",
)
parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="path to config which constructs model",
)
parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="image resolution, in pixel space",
)
parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--center_crop",
    type=bool,
    default=True,
    help="Random Crop"
)
parser.add_argument(
    "--random_flip",
    type=bool,
    default=True,
    help="Random Flip"
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
)
parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
        "--gpus",
        default=[0],
        help="gpu idx",
)
parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
)
parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
)
parser.add_argument(
        '--l_cond',
        default=4,
        type=int,
        help='number of scales'
)
opt = parser.parse_args()

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")
    opt.name = "test" #config['name']

    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device='cuda'
    torch.cuda.set_device(opt.local_rank)

    # dataset
    val_dataset = Custom_dataset(args=opt)
    
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.n_samples,
            shuffle=False,
            num_workers=4,
            pin_memory=False)

    # edge_generator
    net_G = pidinet()
    ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
    net_G.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
    net_G.cuda()

    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)

    # sketch encoder
    model_ad = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
    ckp_ad = torch.load('models/t2iadapter_sketch_sd15v2.pth', map_location='cpu')
    model_ad.load_state_dict({k.replace('module.',''):v for k, v in ckp_ad.items()})

    # fusion block
    FusionNet = fusionnet().to(device)
    ckp_fus = torch.load(opt.model_path, map_location='cpu')
    #ckp_fus = torch.load('./experiments/MaskV2/models/model_fusion_170000.pth', map_location='cpu')
    #ckp_fus = torch.load('./experiments/face_finetune/models/model_fusion_30000.pth', map_location='cpu')
    #ckp_fus = torch.load('./experiments/train_sketch/models/model_fusion_30000.pth', map_location='cpu')
    #ckp_fus = torch.load('./experiments/train_face_MaskV2/models/model_fusion_100000.pth', map_location='cpu')
    FusionNet.load_state_dict(ckp_fus, strict=False)

    # to gpus
    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    FusionNet = torch.nn.parallel.DistributedDataParallel(
        FusionNet,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
        # device_ids=[torch.cuda.current_device()])
    net_G = torch.nn.parallel.DistributedDataParallel(
        net_G,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)


    # testing
    rank, _ = get_dist_info()
    if rank==0:
        for index, data in enumerate(val_dataloader):
            # if index not in [17]:
            #     continue
            with torch.no_grad():
                if opt.dpm_solver:
                    sampler = DPMSolverSampler(model.module)
                elif opt.plms:
                    sampler = PLMSSampler(model.module)
                else:
                    sampler = DDIMSampler(model.module)

                edge = data['sketches'].to(torch.float32).cuda(non_blocking=True)
                
                c = model.module.get_learned_conditioning(data['caption'])
                z1 = model.module.encode_first_stage((data['photos']*2-1.).to(torch.float32).cuda(non_blocking=True))
                z1 = model.module.get_first_stage_encoding(z1)

                if not os.path.exists(os.path.join(opt.data_path, 'visualization')):
                    os.mkdir(os.path.join(opt.data_path, 'visualization'))
                if not os.path.exists(os.path.join(opt.data_path, 'gen_mask')):
                    os.mkdir(os.path.join(opt.data_path, 'gen_mask'))

                
                #features_adapter = model_ad(edge[:,0:1,:,:])
                features_adapter = [model_ad(edge[:,0:1,:,:]), model_ad(edge[:,0:1,:,:]*0)]
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                x_T = None
                
                for i in range(1):
                    samples_ddim, gen_mask = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=model.module.get_learned_conditioning(opt.n_samples * [""]),
                                                eta=opt.ddim_eta,
                                                x_T=x_T,
                                                features_adapter=features_adapter,
                                                fusion_block = FusionNet,
                                                x1 = z1,
                                                cond_tau=1.0,
                                                mask=None)
                                                #mask = data['mask'].to(torch.float32).cuda(non_blocking=True))
                    z1 = samples_ddim
                    
                x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                gen_mask = gen_mask.unsqueeze(1).repeat(1,3,1,1)
                gen_mask = nn.functional.interpolate(gen_mask,scale_factor=8,mode='bilinear')
                #gen_mask = F.max_pool2d(gen_mask, kernel_size=11, stride=1, padding=5)
                #gen_mask= torch.round(gen_mask)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim*gen_mask + (data['photos'].to(torch.float32).cuda(non_blocking=True))*(1-gen_mask)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                for id_sample, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255.* x_sample
                    img = x_sample.astype(np.uint8)
                    cv2.imwrite(os.path.join(opt.data_path, 'visualization', f'{index}.png'), img[:,:,::-1])
                    cv2.imwrite(os.path.join(opt.data_path, 'gen_mask', f'{index}.png'), tensor2img(gen_mask.squeeze(0)))

        
