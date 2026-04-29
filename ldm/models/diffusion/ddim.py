"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               features_adapter=None,
               append_to_context=None,
               cond_tau=1.0,
               style_cond_tau=1.0,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs # fusion_block, x1
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    features_adapter=features_adapter,
                                                    append_to_context=append_to_context,
                                                    cond_tau=cond_tau,
                                                    style_cond_tau=style_cond_tau,**kwargs
                                                    )
        return samples, intermediates
    
    @torch.no_grad()
    def ddim_loop(self, latent, cond_embeddings, t_enc):
        all_latent = [latent]
        all_noise = [None]
        latent = latent.clone().detach()
        timesteps = self.ddim_timesteps[:t_enc]
        total_steps = timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(timesteps, desc='Encoding image', total=total_steps)
        for i, step in enumerate(iterator):
            ts = torch.full((latent.shape[0],), step,
                            device=latent.device, dtype=torch.long)
            # print(ts, step)  # 1, 21, 41....981
            #noise_pred, _ = self.model.apply_model(latent, ts, cond_embeddings)
            noise_pred = self.model.apply_model(latent, ts, cond_embeddings)
            latent = self.next_step(
                noise_pred, step, latent, use_original_steps=True)
            all_latent.append(latent)
            all_noise.append(noise_pred)
        return all_latent, all_noise
    
    @torch.no_grad()
    def ddim_loop_stroke(self, latent, cond_embeddings, t_enc, fusion_block, features_adapter):
        kwargs1 = {'features_adapter':None, 'mid_out': True}
        t = torch.full((latent.shape[0],), 1,
                            device=latent.device, dtype=torch.long) * 273
        _, ori_img_features = self.model.apply_model(self.model.q_sample(latent, t),
                                                      t, cond_embeddings,return_all=True, **kwargs1)
        features_adapter_fuse = fusion_block(features_adapter, ori_img_features)

        all_latent = [latent]
        all_noise = [None]
        latent = latent.clone().detach()
        timesteps = self.ddim_timesteps[:t_enc]
        total_steps = timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(timesteps, desc='Encoding image', total=total_steps)
        for i, step in enumerate(iterator):
            ts = torch.full((latent.shape[0],), step,
                            device=latent.device, dtype=torch.long)
            # print(ts, step)  # 1, 21, 41....981
            #noise_pred, _ = self.model.apply_model(latent, ts, cond_embeddings)
            noise_pred = self.model.apply_model(latent, ts, cond_embeddings,features_adapter=features_adapter_fuse)
            latent = self.next_step(
                noise_pred, step, latent, use_original_steps=True)
            all_latent.append(latent)
            all_noise.append(noise_pred)
        return all_latent, all_noise

    def next_step(self, e_t, timestep, x, use_original_steps=False, quantize_denoised=False):
        next_timestep = min(timestep + 1000 // len(self.ddim_timesteps), 999)
        b, *_, device = *x.shape, x.device
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # select parameters corresponding to the currently considered timestep
        a_next = torch.full((b, 1, 1, 1), alphas[next_timestep], device=device)
        a_t = torch.full((b, 1, 1, 1), alphas[timestep], device=device)
        sqrt_one_minus_a_t = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[timestep], device=device)
        next_x0 = (x - sqrt_one_minus_a_t * e_t) / a_t.sqrt()
        # current prediction for x_0
        if quantize_denoised:
            next_x0, _, *_ = self.model.first_stage_model.quantize(next_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * next_x0 + dir_xt
        return x_next

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, features_adapter=None,
                      append_to_context=None, cond_tau=0.4, style_cond_tau=1.0,fusion_block = None, x1 = None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        #DDIM inversion
        assert x1 is not None
        ref_latents, ref_noises = self.ddim_loop_stroke(x1, cond, len(timesteps),
                                                        features_adapter=features_adapter[1],fusion_block=fusion_block)
        ref_latents = list(reversed(ref_latents))
        ref_noises = list(reversed(ref_noises))
        generate_mask = None
        #DDIM reverse Loop
        lamda = 0.0
        cmask = 0
        self.mask_gen = None
        self.features_adapter_fuse = None
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            img_orig = self.model.q_sample(x1, torch.ones_like(ts)*273)
            warning_step = 40
            if i==0:
                # initial x_T from x_hat_0
                img = ref_latents[0]
            elif i!=0 and i<warning_step:
                cmask = generate_mask.squeeze(1)
                img = img * cmask[:,None,:,:] + (1-cmask[:,None,:,:])*(ref_latents[i])
            else:
                cmask = generate_mask.squeeze(1)
                img = img * cmask[:,None,:,:] + (1-cmask[:,None,:,:])*(ref_latents[i])
            
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      features_adapter=None if index < int(
                                          (1 - cond_tau) * total_steps) else features_adapter[0],
                                      append_to_context=None if index < int(
                                          (1 - style_cond_tau) * total_steps) else append_to_context,
                                          fusion_block = fusion_block, img_ori =img_orig,
                                          ref_noise= ref_noises[i],lamda=lamda
                                      )
            # img, pred_x0 = outs
            img, pred_x0, generate_mask = outs
            if mask is not None:
                generate_mask = mask
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        generate_mask = generate_mask.squeeze(1)
        img = img * generate_mask[:,None,:,:] + (1-generate_mask[:,None,:,:])*x1
        return img, generate_mask #intermediates
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, features_adapter=None,
                      append_to_context=None, fusion_block = None, img_ori =None, ref_noise=None, lamda=0.15):
        b, *_, device = *x.shape, x.device

        if features_adapter is not None:
            if self.features_adapter_fuse == None:
                kwargs1 = {'features_adapter':None, 'mid_out': True}
                _, ori_img_features = self.model.apply_model(img_ori, torch.ones_like(t)*273, c,return_all=True, **kwargs1)
                features_adapter_fuse = fusion_block(features_adapter, ori_img_features)
                mask_gen = torch.round(features_adapter_fuse[0][1])
                self.mask_gen = mask_gen
                self.features_adapter_fuse = features_adapter_fuse
            else:
                features_adapter_fuse = self.features_adapter_fuse
                mask_gen = self.mask_gen
            
            #mask_gen = (features_adapter_fuse[0][1]>0.5).float()
        else:
            features_adapter_fuse = None

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            if append_to_context is not None:
                model_output = self.model.apply_model(x, t, torch.cat([c, append_to_context], dim=1),
                                                      features_adapter=features_adapter_fuse)
            else:
                model_output = self.model.apply_model(x, t, c, features_adapter=features_adapter_fuse)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if features_adapter_fuse is not None:
                features_adapter_fuse = [
                    [torch.cat([features_adapter_fuse[i][0]]*2),
                     torch.cat([features_adapter_fuse[i][1]]*2),
                     #torch.cat([features_adapter_fuse[i][2]]*2)
                     ]
                    for i in range(len(features_adapter_fuse))
                                          ]
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                if append_to_context is not None:
                    pad_len = append_to_context.size(1)
                    new_unconditional_conditioning = torch.cat(
                        [unconditional_conditioning, unconditional_conditioning[:, -pad_len:, :]], dim=1)
                    new_c = torch.cat([c, append_to_context], dim=1)
                    c_in = torch.cat([new_unconditional_conditioning, new_c])
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, features_adapter=features_adapter_fuse).chunk(2)
            
            if ref_noise is not None:
                # inspired by proximal npi
                # tmp = torch.zeros_like(model_t)
                # lamda = lamda
                # # tmp[(model_t-ref_noise)>lamda] = model_t[(model_t-ref_noise)>lamda]#-lamda
                # # tmp[(model_t-ref_noise)<-lamda] = model_t[(model_t-ref_noise)<-lamda]#+lamda
                # tmp[(model_t-ref_noise)>lamda] = -lamda
                # tmp[(model_t-ref_noise)<-lamda] = lamda
                # mask = (tmp!=0).float()
                #model_output = model_t #+  tmp#(model_t - ref_noise)
                model_output = model_uncond + 3.5 * (model_t - model_uncond)
                #model_output = model_t+tmp
                
                # proximal-npi  
                # mask = 1-(tmp!=0).float()
                # model_output = ref_noise + unconditional_guidance_scale * tmp      
            else:
                mask = None
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise_tmp = noise_like(x.shape, device, repeat_noise)
        noise = sigma_t * noise_tmp * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # return x_prev, pred_x0
        return x_prev, pred_x0, mask_gen #mask

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
