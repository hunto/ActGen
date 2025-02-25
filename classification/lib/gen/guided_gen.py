import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from diffusers import StableDiffusionPipeline
from lib.gen.DDPMScheduler_G import DDPMScheduler_G
from diffusers.image_processor import VaeImageProcessor
from .attn import AttentionStore, AttnProcessor
# from torchvision.models import resnet18

from lib.dataset.transform import CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from lib.dataset.categories import CIFAR10_CATEGORIES, CIFAR100_CATEGORIES, IMAGENET_SIMPLE_CATEGORIES
import time
import copy
import torch.distributed as dist


class Normalize:
    def __init__(self, prev_mean=CIFAR_DEFAULT_MEAN, prev_std=CIFAR_DEFAULT_STD, tgt_mean=0.5, tgt_std=0.5):
        if not isinstance(tgt_mean, (tuple, list)):
            tgt_mean = [tgt_mean, ] * 3
        if not isinstance(tgt_std, (tuple, list)):
            tgt_std = [tgt_std, ] * 3
        # new_x = (x * prev_std + prev_mean - tgt_mean) / tgt_std
        #       = x * prev_std / tgt_std + (prev_mean - tgt_mean) / tgt_std
        self.mul = torch.Tensor([prev / tgt for prev, tgt in zip(prev_std, tgt_std)]).view(1, 3, 1, 1).cuda()
        self.add = torch.Tensor([(pm - tm) / ts for pm, tm, ts in zip(prev_mean, tgt_mean, tgt_std)]).view(1, 3, 1, 1).cuda()

    def __call__(self, x):
        x = x * self.mul + self.add
        return x

class DeNormalize:
    def __init__(self, prev_mean=CIFAR_DEFAULT_MEAN, prev_std=CIFAR_DEFAULT_STD):
        self.prev_mean = torch.Tensor(prev_mean).view(1, 3, 1, 1).cuda()
        self.prev_std = torch.Tensor(prev_std).view(1, 3, 1, 1).cuda()
    
    def __call__(self, x):
        x = x * self.prev_std + self.prev_mean
        return x


class AdvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, target):
        loss = self.loss(logits, target)
        # loss = torch.where(loss > 6, torch.zeros(loss.shape, device=loss.device), loss)
        # probs = logits.softmax(-1)
        # probs = torch.where(F.one_hot(target, num_classes=probs.shape[1]) == 1, probs - 0.1, probs)
        # mask = probs.argmax(1) == target
        # loss = torch.where(mask, loss, torch.zeros(loss.shape, device=loss.device))
        return loss.mean()

class GuidedGen:
    def __init__(
        self,
        guidance_str = 12.5,
        guide_lambda=0.005,
        pretrained_model_name_or_path='/mnt/afs/huangtao3/models/stable-diffusion-2-1-base-save',
        dataset='cifar10'
    ):
        if dataset == 'cifar10':
            self.categories = CIFAR10_CATEGORIES
            prev_mean, prev_std = CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD
            self.model_input_size = 32
        elif dataset == 'cifar100':
            self.categories = CIFAR100_CATEGORIES
            prev_mean, prev_std = CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD
            self.model_input_size = 32
        elif dataset == 'imagenet':
            self.categories = IMAGENET_SIMPLE_CATEGORIES
            prev_mean, prev_std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            self.model_input_size = 224

        self.weight_type = torch.float16

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=self.weight_type,
            # variant="fp16",
        ).to('cuda')


        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.vae.to(self.weight_type)
        self.pipeline.text_encoder.to(self.weight_type)
        self.pipeline.unet.to(self.weight_type)
        # self.ori_scheduler = self.pipeline.scheduler
        self.guided_scheduler = DDPMScheduler_G.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weight_type,
            # variant='fp16'
        )
        self.vae_processor = VaeImageProcessor()
        # self.pipeline.enable_vae_slicing()
        # self.pipeline.enable_vae_tiling()
        # self.pipeline.unet.enable_xformers_memory_efficient_attention()

        self.guide_lambda = guide_lambda
        self.guidance_str = guidance_str

        # transform to normalize images to sd format
        self.normalize = Normalize(prev_mean, prev_std)
        self.normalize_sd2cls = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], prev_mean, prev_std)

        attn_res = int(np.ceil(512 / 32)), int(np.ceil(512 / 32))
        self.attention_store = AttentionStore(attn_res)

        attn_procs = {}
        cross_att_count = 0
        for name in self.pipeline.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.pipeline.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

        self.encoded_prompts = {}
        self.model = None
        self.mem_bank = {}
        self.rand_gen_ratio = 0.

    
    def cal_contrastive_loss(self, ori_latent, label, margin=100):
        # return 0
        n_items = 1024
        if label.item() not in self.mem_bank:
            return 0
        feats = self.mem_bank[label.item()]
        n_items = min(n_items, len(feats))
        feats = random.sample(feats, n_items)
        feats = torch.stack(feats).view(n_items, -1).float().cuda()
        ori_latent = ori_latent.view(1, -1).float()
        # cal distance
        euc = (feats - ori_latent)**2
        euc = euc.sum(1)
        euc = torch.sqrt(euc)
        loss = (margin - euc).clamp(0).mean()
        return loss

    def __call__(self, images, labels, model):
        """
        @images: torch.Tensor [B, 3, H, W], images generated & normalized by the dataset
        """
        if self.model is None:
            self.model = copy.deepcopy(model.module)
            self.model.requires_grad_(False)
        else:
            self.model.load_state_dict(model.module.state_dict())

        t = time.time()
        noise_scheduler = self.guided_scheduler

        num_inference_steps = 40
        device = torch.device('cuda')
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps
        images = self.normalize(images)
        images = F.interpolate(images, (512,512), mode='bilinear').to(self.weight_type)
        # base = math.e

        # print('normalize', time.time() - t)
        t = time.time()

        with torch.no_grad():
            # guided_latents = self.pipeline.vae.encode(self.vae_processor.preprocess(images)).latent_dist.sample()
            guided_latents = self.pipeline.vae.encode(images).latent_dist.sample()
            guided_latents = self.pipeline.vae.config.scaling_factor * guided_latents

        # print('encode', time.time() - t)
        t = time.time()

        # model = resnet18(num_classes=len(self.categories), weights=None).to(device)
        # # model.load_state_dict(torch.load(model_path))
        # model.load_state_dict(torch.load('/content/cifar10.pth'))
        self.model.eval()
        # model.requires_grad_(False)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = AdvLoss()

        to_pil = transforms.ToPILImage()
        gen_images = []
        for idx, (guided_latent, label) in enumerate(zip(guided_latents, labels)):
            rand_gen = random.random() < self.rand_gen_ratio
            if rand_gen:
                prompt = f'a photo of {self.categories[label]}'
                latents = self.pipeline(
                    prompt,
                    height=512,
                    width=512,
                    num_inference_steps=40,
                    output_type='latent',
                    return_dict=True
                ).images[0:1]
                gen_images.append(latents)
                continue

            # self.guidance_str = random.random() * 15
            # print(f'Now generating: {idx}/{len(guided_latents)}')
            if label.item() not in self.encoded_prompts:
                prompt = f'a photo of {self.categories[label]}'
                # print(label, self.categories[label])

                prompt_embeds = self.pipeline._encode_prompt(
                    prompt,
                    device,
                    1,
                    do_classifier_free_guidance=True,
                )
                self.encoded_prompts[label.item()] = prompt_embeds.cpu()
            else:
                prompt_embeds = self.encoded_prompts[label.item()].cuda()
            prompt_embeds = prompt_embeds.clone()

            # print('encode prompt', time.time() - t)
            t = time.time()

            # timestep = torch.LongTensor([0]).to(device)
            height = width = 512 # pipeline.unet.config.sample_size * pipeline.vae_scale_factor

            latents = self.pipeline.prepare_latents(
                1,
                self.pipeline.unet.in_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator=None
            )

            latents_guide = self.pipeline.prepare_latents(
                1,
                self.pipeline.unet.in_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator=None
            )

            t = time.time()

            attn_mask = torch.full((16, 16), 1).to(device)
            self.attention_store.reset()

            pred_label = label
            target = torch.LongTensor([label]).cuda()
            use_adv = random.random() > self.rand_gen_ratio + 0.5
            self.pipeline.unet.enable_gradient_checkpointing()
            for i, timestep in enumerate(timesteps):
                self.pipeline.unet.zero_grad()
                if i < 10:
                    prompt_embeds.requires_grad_(True)
                else:
                    prompt_embeds.requires_grad_(False)

                prompt_embeds = prompt_embeds.to(self.weight_type)
                # with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2)
                noise_pred = self.pipeline.unet(latent_model_input, timestep, prompt_embeds, return_dict=True).sample
                # classifier-free guidance
                guidance_scale = 7.5
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                attn_mask = torch.stack(
                    [self.attention_store.aggregate_attention(from_where=['up', 'down'])[:, :, 4], attn_mask],
                    dim=0)
                attn_mask = torch.sum(attn_mask, dim=0)
                latents_attn_mask = attn_mask.unsqueeze(0).expand(4, *attn_mask.shape).unsqueeze(0)
                latents_attn_mask = F.interpolate(latents_attn_mask, (64, 64), mode='bilinear')

                latents.requires_grad_(True)
                output = noise_scheduler.step(noise_pred, timestep, latents, guided_latent, latents_guide)
                latents_origin = output.pred_original_sample

                if i < 10:
                    ce_grad = 0
                    if use_adv:
                        latents_convert = 1 / 0.18215 * latents_origin
                        latents_convert = self.pipeline.vae.decode(latents_convert).sample
                        latents_convert = (latents_convert / 2 + 0.5).clamp(0, 1)
                        latents_convert = F.interpolate(latents_convert, (self.model_input_size, self.model_input_size), mode='bilinear')
                        latents_convert = self.normalize_sd2cls(latents_convert)
                        latents_convert = latents_convert.to(torch.float)
                        outputs = self.model(latents_convert)
                        ce_loss = criterion(outputs, target) * 0.1
                        if dist.get_rank() == 0:
                            print('ce_loss', ce_loss)
                        preds = F.softmax(outputs, -1)
                        max_value, max_index = preds.max(dim=1)
                        pred_label = max_index.item()
                        t = time.time()
                    else:
                        ce_loss = 0

                    contra_loss = self.cal_contrastive_loss(latents_origin, label)
                    if contra_loss != 0:
                        print('contra_loss: ', contra_loss)
                    loss = contra_loss - ce_loss
                else:
                    loss = 0
                if not isinstance(loss, torch.Tensor):
                    grad = torch.zeros(prompt_embeds.shape, device=prompt_embeds.device)
                else:
                    grad =  torch.autograd.grad(
                        outputs=loss,
                        inputs=prompt_embeds,
                        create_graph=False,
                        retain_graph=False
                    )[0]
                    grad = F.normalize(grad)
                    grad[torch.isnan(grad)] = 0.0

                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                util = int(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
                mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024

                latents = output.prev_sample.detach()
                latents_guide = output.prev_sample_guide.detach()
                perturb_rate = 1 - (math.e ** (i - self.guidance_str) / (1 + math.e ** (i - self.guidance_str)))
                latents = latents - perturb_rate * latents_attn_mask.detach() * (latents - latents_guide)
                latents = latents.to(self.weight_type).detach()
                prompt_embeds = prompt_embeds.detach() - 0.05 * grad.detach()
                # print('grad', time.time() - t)
                t = time.time()
            image = latents.detach()
            if label.item() in self.mem_bank:
                self.mem_bank[label.item()].append(image.cpu())
            else:
                self.mem_bank[label.item()] = [image.cpu()]
            gen_images.append(image)
            # print('decode', time.time() - t)
            t = time.time()
        gen_images_ = torch.cat(gen_images, 0)
        gen_images_ = self.pipeline.decode_latents(gen_images_)
        gen_images_ = self.pipeline.numpy_to_pil(gen_images_)
        return gen_images_


    def pred_original_sample(self, sample, pred_noise, timestep):
        alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[timestep].reshape(-1, 1, 1, 1).to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        origin = (sample - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
        return origin


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
    ])
    train_dataset = datasets.CIFAR10(
        root='data/cifar', train=True, download=True, transform=test_transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
    model_path = ''
    images, labels = next(iter(dataloader))
    images = images.cuda()

    gg = GuidedGen()

    gen_images = gg(images, labels, model_path)
    dn = DeNormalize()
    images = dn(images)
    tensor2pil = transforms.ToPILImage()
    images = [tensor2pil(img) for img in images]
    print(len(images), len(gen_images))
    for idx, (ori, gen) in enumerate(zip(images, gen_images)):
        ori.save(f'imgs/{idx}_ori.png')
        gen.save(f'imgs/{idx}_gen.png')