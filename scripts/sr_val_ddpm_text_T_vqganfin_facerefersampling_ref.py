"""make variations of input image"""

import argparse, os, sys, glob
import cv2
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

cur_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(cur_dir)
print(sys.path)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from basicsr.metrics import calculate_niqe
from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor
import math
import copy


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print('>>>>>>>>>>>>>>>>>>>load results>>>>>>>>>>>>>>>>>>>>>>>')
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def load_img_lq(config, path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0

    opt = config.test_data.params
    blur_kernel_size = opt['blur_kernel_size']
    kernel_list = opt['kernel_list']
    kernel_prob = opt['kernel_prob']
    blur_sigma = opt['blur_sigma']
    downsample_range = opt['downsample_range']
    noise_range = opt['noise_range']
    jpeg_range = opt['jpeg_range']
    kernel = degradations.random_mixed_kernels(
        kernel_list,
        kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma, [-math.pi, math.pi],
        noise_range=None)
    img_lq = cv2.filter2D(image, -1, kernel)
    # downsample
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
    # noise
    if noise_range is not None:
        img_lq = degradations.random_add_gaussian_noise(img_lq, noise_range)
    # jpeg compression
    if jpeg_range is not None:
        img_lq = degradations.random_add_jpg_compression(img_lq, jpeg_range)

    # resize to original size
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
    # img_lq = img2tensor(img_lq, bgr2rgb=False, float32=True)
    # img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

    image = image[None].transpose(0, 3, 1, 2)
    img_lq = img_lq[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    img_lq = torch.from_numpy(img_lq)
    img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
    return 2.*image - 1., 2.*img_lq - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="/mnt/lustre/share/jywang/dataset/ImageSR/RealSRSet/"
    )
    parser.add_argument(
        "--ref-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="/mnt/lustre/share/jywang/dataset/ImageSR/RealSRSet/"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/sr-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=1.0,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--save_input",
        action='store_true',
        help="if enabled, save inputs",
    )
    parser.add_argument(
        "--facesr",
        action='store_true',
        help="if enabled, save inputs",
    )
    parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
    parser.add_argument(
        "--gen_lq",
        action='store_true',
        help="if enabled, generate lq",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model = model.to(device)

    if opt.facesr:
        vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi_face.yaml")
        vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    else:
        vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
        vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)

    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    input_path = os.path.join(outpath, "inputs")
    os.makedirs(input_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    base_i = len(os.listdir(input_path))
    grid_count = len(os.listdir(outpath)) - 1

    img_list_old = os.listdir(opt.init_img)
    img_list_renew = []
    save_list = os.listdir(sample_path)
    for item in img_list_old:
        if item in save_list:
            pass
        else:
            img_list_renew.append(item)

    img_list = sorted(img_list_renew)
    niters = math.ceil(len(img_list) / batch_size)
    # using list comprehension
    img_list_chunk = [img_list[i * batch_size:(i + 1) * batch_size] for i in range((len(img_list) + batch_size - 1) // batch_size )]

    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    model_ori = copy.deepcopy(model)

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)
    model_ori = model_ori.to(device)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    niqe_list = []
    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(niters, desc="Sampling"):
                cur_img_list = img_list_chunk[n]
                init_image_list = []
                ref_image_list = []
                for item in cur_img_list:
                    if opt.gen_lq:
                        _, cur_image = load_img_lq(config, os.path.join(opt.init_img, item))
                        cur_image = cur_image.to(device)
                    else:
                        cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
                    cur_ref_image = load_img(os.path.join(opt.ref_img, item)).to(device)
                    cur_image = transform(cur_image)
                    cur_ref_image = transform(cur_ref_image)
                    init_image_list.append(cur_image)
                    ref_image_list.append(cur_ref_image)
                init_image = torch.cat(init_image_list, dim=0)
                ref_image = torch.cat(ref_image_list, dim=0)
                init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
                init_latent = model.get_first_stage_encoding(init_latent_generator)

                # ref_latent_generator, _ = vq_model.encode(ref_image)
                # ref_latent = model.get_first_stage_encoding(ref_latent_generator)

                text_init = ['']*init_image.size(0)
                print(model.device, model.cond_stage_model.device)
                print(device)
                model.cond_stage_model = model.cond_stage_model.to(device)
                print(model.device, model.cond_stage_model.device)

                null_cond = model.cond_stage_model(text_init)


                cond = -1
                text_init = ['a younger portrait of *' if cond == -1 else 'an older portrait of *'] * init_image.size(0)
                text_cond = model.cond_stage_model(text_init)

                noise = torch.randn_like(init_latent)
                t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                t = t.to(device).long()
                x_T = model_ori.q_sample(x_start=init_latent, t=t, noise=noise)
                # x_T = None

                samples, _ = model.sample(text_cond=text_cond, null_cond=null_cond, struct_cond=init_latent, ref=ref_image, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, interfea_path=None,
                                        reference_sr=None, reference_lr=0.5, reference_step=10, reference_range=[10, 1000])
                # _, enc_fea_lq = vq_model.encode(init_image)
                x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                # x_samples = vq_model.decode(samples, enc_fea_lq)
                # x_samples = model.decode_first_stage(samples)
                # x_samples = adaptive_instance_normalization(x_samples, init_image)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                if not opt.skip_save:
                    for i in range(x_samples.size(0)):
                        x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                        niqe_list.append(calculate_niqe(x_sample, 0, input_order='HWC', convert_to='y'))
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, cur_img_list[i]))
                    if opt.save_input:
                        for i in range(init_image.size(0)):
                            x_input = 255. * rearrange(init_image[i].cpu().numpy(), 'c h w -> h w c')
                            x_input = (x_input+255.)/2
                            Image.fromarray(x_input.astype(np.uint8)).save(
                                os.path.join(input_path, cur_img_list[i]))
                    base_i += init_image.size(0)

            if not opt.skip_grid:
                # additionally, save as grid
                all_samples_new = []
                for item in all_samples:
                    if item.size(0) < batch_size:
                        template_tensor = item[0].unsqueeze(0)
                        add_tensor = torch.zeros_like(template_tensor).repeat(batch_size-item.size(0), 1,1,1)
                        item = torch.cat([item, add_tensor], dim=0)
                        assert item.size(0) == batch_size
                    all_samples_new.append(item)
                grid = torch.stack(all_samples_new, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

            assert len(niqe_list) == len(img_list)
            avg_niqe = np.mean(np.array(niqe_list))

            print(f"Average NIQE score: {avg_niqe:.3f} \n")

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
