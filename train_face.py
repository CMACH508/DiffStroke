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

from dist_util import get_bare_model, get_dist_info, init_dist, master_only
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.model_edge import pidinet
from ldm.modules.fusion_block import fusionnet
from Dataset import Face_data
from torchvision.transforms import Resize

os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='1234'

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

@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))

def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        # check_resume(opt, resume_state['iter'])
    return resume_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=4,
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
        default='./Dataset/CelebA-HQ',
        help="path of dataset",
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
        default=2,
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
        default=7.5,
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
    opt.name = config['name']

    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device='cuda'
    torch.cuda.set_device(opt.local_rank)

    # dataset
    train_dataset = Face_data(args=opt, mode='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_dataset = Face_data(args=opt, mode='test')

    # train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=opt.bsize,
    #         shuffle=True,
    #         num_workers=opt.num_workers,
    #         pin_memory=True)

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.bsize,
            shuffle=(train_sampler is None),
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.n_samples,
            shuffle=False,
            num_workers=1,
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
    model_ad.eval()
    model_ad.requires_grad = False
    net_G.requires_grad = False
    model.requires_grad = False
    # fusion block
    FusionNet = fusionnet().to(device)
    ckp_fn = torch.load('./experiments/MaskV2/models/model_fusion_170000.pth', map_location='cpu')
    FusionNet.load_state_dict({k.replace('module.',''):v for k, v in ckp_fn.items()})

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
        # device_ids=[torch.cuda.current_device()])

    # optimizer
    params = list(FusionNet.parameters())
    # optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])
    optimizer = torch.optim.AdamW(params, 1e-4)

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)


    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    resize = Resize((opt.resolution,opt.resolution))
    for epoch in range(start_epoch, opt.epochs):
        # val
        rank, _ = get_dist_info()
        # if rank==0:
            # for data in val_dataloader:
            #     with torch.no_grad():
            #         if opt.dpm_solver:
            #             sampler = DPMSolverSampler(model.module)
            #         elif opt.plms:
            #             sampler = PLMSSampler(model.module)
            #         else:
            #             sampler = DDIMSampler(model.module)

            #         edge1 = net_G(data['pixel_value'].cuda(non_blocking=True))[-1]
            #         edge2 = net_G(data['deformed_value'].cuda(non_blocking=True))[-1]
            #         edge1 = edge1>0.2
            #         edge2 = edge2>0.2
            #         edge = resize((edge2.float()).clip(min=0)*(data['deformed_mask'].to(torch.float32).cuda(non_blocking=True)))

            #         c = model.module.get_learned_conditioning(data['caption'])
            #         z1 = model.module.encode_first_stage(resize((data['pixel_value']*2-1.)).to(torch.float32).cuda(non_blocking=True))
            #         z1 = model.module.get_first_stage_encoding(z1)
            #         z2 = model.module.encode_first_stage(resize((data['deformed_value']*2-1.)).to(torch.float32).cuda(non_blocking=True))
            #         z2 = model.module.get_first_stage_encoding(z2)

            #         cv2.imwrite(os.path.join(experiments_root, 'visualization', 'edge_%04d.png'%epoch), tensor2img(1-edge))
            #         cv2.imwrite(os.path.join(experiments_root, 'visualization', 'ori_img_%04d.png'%epoch), tensor2img(data['pixel_value']))
            #         cv2.imwrite(os.path.join(experiments_root, 'visualization', 'gt_img_%04d.png'%epoch), tensor2img(data['deformed_value']))
            #         features_adapter = [model_ad(edge[:,0:1,:,:]), model_ad(edge[:,0:1,:,:]*0)]

            #         shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            #         x_T = None

            #         samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
            #                                             conditioning=c,
            #                                             batch_size=opt.n_samples,
            #                                             shape=shape,
            #                                             verbose=False,
            #                                             unconditional_guidance_scale=opt.scale,
            #                                             unconditional_conditioning=model.module.get_learned_conditioning(opt.n_samples * [""]),
            #                                             eta=opt.ddim_eta,
            #                                             x_T=x_T,
            #                                             features_adapter=features_adapter,
            #                                             fusion_block = FusionNet,
            #                                             x1 = z1)
                    
            #         x_samples_ddim = model.module.decode_first_stage(samples_ddim)
            #         x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            #         x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            #         for id_sample, x_sample in enumerate(x_samples_ddim):
            #             x_sample = 255.* x_sample
            #             img = x_sample.astype(np.uint8)
            #             cv2.imwrite(os.path.join(experiments_root, 'visualization', 'sample_e%04d_s%04d.png'%(epoch, id_sample)), img[:,:,::-1])
            #         break

        train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in enumerate(train_dataloader):
            current_iter += 1
            with torch.no_grad():
                edge1 = net_G(data['pixel_value'].cuda(non_blocking=True))[-1]
                edge2 = net_G(data['deformed_value'].cuda(non_blocking=True))[-1]
                edge1 = resize(edge1>0.2)
                edge2 = resize(edge2>0.2)
                rough_mask = resize(data['deformed_mask'].to(torch.float32).cuda(non_blocking=True))
                edge = (edge2.float()-edge1.float()).clip(min=0)*(resize(data['deformed_mask'].to(torch.float32).cuda(non_blocking=True)))

                c = model.module.get_learned_conditioning(data['caption'])
                z1 = model.module.encode_first_stage(resize((data['pixel_value']*2-1.)).to(torch.float32).cuda(non_blocking=True))
                z1 = model.module.get_first_stage_encoding(z1)
                z2 = model.module.encode_first_stage(resize((data['deformed_value']*2-1.)).to(torch.float32).cuda(non_blocking=True))
                z2 = model.module.get_first_stage_encoding(z2)

            optimizer.zero_grad()
            model.zero_grad()

            features_adapter = model_ad(edge[:,0:1,:,:])
            features_adapter_gt = model_ad(edge2[:,0:1,:,:].float().clip(min=0))

            # [condition, target]
            z = [z1,z2]
            rough_mask = nn.functional.interpolate(rough_mask,scale_factor=1/8, mode='bilinear')
            l_pixel, loss_dict = model(z, c=c, fusion_block= FusionNet, features_adapter=features_adapter, 
                                       features_adapter_gt=features_adapter_gt, 
                                       rough_mask=rough_mask[:,:1,:,:])
            l_pixel.backward()
            optimizer.step()

            #if (current_iter+1)%opt.print_fq == 0:
            if (current_iter+1)%20 == 0:
                logger.info(loss_dict)

            # save checkpoint
            rank, _ = get_dist_info()
            if ((rank==0) and ((current_iter+1)%config['training']['save_freq'] == 0)):
                print("saving model")
                # save_filename = f'model_ad_{current_iter+1}.pth'

                save_filename = f'model_fusion_{current_iter+1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                state_dict_fu = {}
                FusionNet_bare = get_bare_model(FusionNet)
                state_dict_fu = FusionNet_bare.state_dict()
                for key, param in state_dict_fu.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict_fu[key] = param.cpu()
                torch.save(state_dict_fu, save_path)

            # save state
                state = {'epoch': epoch, 'iter': current_iter+1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter+1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)

        
