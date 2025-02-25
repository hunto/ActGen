import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, \
    SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, \
    CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops
from lib.dataset.utils import split_cifar_dataset, split_imagenet_dataset
from lib.dataset.dataloader import DataPrefetcher
from lib.dataset.builder import _LOADER_PARAMS
from lib.dataset.utils import extend_cifar_dataset, extend_imagenet_dataset


try:
    # need `pip install nvidia-ml-py3` to measure gpu stats
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    _has_nvidia_smi = True
except ModuleNotFoundError:
    _has_nvidia_smi = False


torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

FILE_CACHE_PATH = '/mnt/afs/huangtao3/icsota_gen/data/saved_imgs'

from lib.dataset.categories import CIFAR10_CATEGORIES, CIFAR100_CATEGORIES
from PIL import Image

def read_images():
    labels = []
    images = []
    dataset_path = 'data/rg/cifar10'
    for cate in os.listdir(dataset_path):
        print(cate)
        lbl = CIFAR10_CATEGORIES.index(cate)
        path = dataset_path + '/' + cate
        for img in os.listdir(path):
            img = np.array(Image.open(path + '/' + img).convert('RGB'))
            images.append(img)
            labels.append(lbl)
    images = np.stack(images)
    labels = np.array(labels)
    return images, labels

class ImageToken(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.size = size
        self.param = nn.Parameter(torch.randn(1, 3, size, size).normal_(0, 0.001), requires_grad=True)

    def forward(self, x, mask):
        out = torch.where(mask.view(-1, 1, 1, 1).expand(-1, 3, self.size, self.size), x + self.param, x)
        return out
        x[mask] = x[mask] + self.param.weight
        return x


def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    # save args
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''build dataloader'''
    train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)

    # print(len(train_dataset))
    # images, labels = read_images()
    # extend_cifar_dataset(train_dataset, images, labels)
    # sampler = train_loader.loader.sampler.__class__(train_dataset, shuffle=train_loader.loader.sampler.shuffle)
    # dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
    #     pin_memory=False, sampler=sampler, collate_fn=train_loader.loader.collate_fn, drop_last=True, **_LOADER_PARAMS)
    # train_loader.loader = dataloader
    # print(len(train_dataset))

    if args.gen_images:
        # split dataset
        if args.dataset in ('cifar10', 'cifar100'):
            train_dataset, gen_val_dataset = split_cifar_dataset(train_dataset, 0.9)
        elif args.dataset == 'imagenet':
            train_dataset, gen_val_dataset = split_imagenet_dataset(train_dataset, 1 - 10000 / len(train_dataset))
        else:
            raise RuntimeError(f'Dataset type <{args.dataset}> is not supported.')

        sampler = train_loader.loader.sampler.__class__(train_dataset, shuffle=train_loader.loader.sampler.shuffle)
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
            pin_memory=False, sampler=sampler, collate_fn=train_loader.loader.collate_fn, drop_last=True, **_LOADER_PARAMS)
        train_loader.loader = dataloader # = DataPrefetcher(train_loader, train_transforms_r, mixup_transform)

        gen_val_dataset.transform = val_dataset.transform
        sampler = val_loader.loader.sampler.__class__(gen_val_dataset, shuffle=val_loader.loader.sampler.shuffle)
        dataloader = torch.utils.data.DataLoader(
            gen_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
            pin_memory=False, sampler=sampler, collate_fn=val_loader.loader.collate_fn, drop_last=False, **_LOADER_PARAMS)
        gen_val_loader = DataPrefetcher(dataloader, val_loader.transforms, mixup_transform=val_loader.mixup_transform)
    ori_size = len(train_dataset)

    '''build model'''
    if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
        loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                          epsilon=args.smoothing).cuda()
    val_loss_fn = loss_fn

    model = build_model(args, args.model)
    logger.info(model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    # Diverse Branch Blocks
    if args.dbb:
        # convert 3x3 convs to dbb blocks
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info(model)
        logger.info(
            f'Converted to DBB blocks, model params: {get_params(model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    # model.add_module('_image_token', ImageToken())
    model.cuda()
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)

    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
        teacher_model.cuda()
        test_metrics = validate(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        # build kd loss
        from lib.models.losses.kd_loss import KDLoss
        loss_fn = KDLoss(model, teacher_model, loss_fn, args.kd, args.student_module,
                         args.teacher_module, args.ori_loss_weight, args.kd_loss_weight)

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''build optimizer'''
    optimizer = build_optimizer(args.opt,
                                model.module,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)

    '''build scheduler'''
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(args.sched,
                                optimizer,
                                warmup_steps,
                                args.warmup_lr,
                                decay_steps,
                                args.decay_rate,
                                total_steps,
                                steps_per_epoch=steps_per_epoch,
                                decay_by_epoch=args.decay_by_epoch,
                                min_lr=args.min_lr)

    '''dyrep'''
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        from lib.models.utils.recal_bn import recal_bn
        dyrep = DyRep(
            model.module,
            optimizer,
            recal_bn_fn=lambda m: recal_bn(model.module, train_loader,
                                           args.dyrep_recal_bn_iters, m),
            filter_bias_and_bn=not args.opt_no_filter)
        logger.info('Init DyRep done.')
    else:
        dyrep = None

    '''amp'''
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'scaler': loss_scaler,
                                         'dyrep': dyrep
                                     })

    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if start_epoch > args.warmup_epochs:
            scheduler.finished = True
        scheduler.step(start_epoch * len(train_loader))
        if args.dyrep:
            model = DDP(model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'start training from epoch {start_epoch}'
        )

        # resume saved images
        if args.dataset == 'imagenet':
            pass
        start_epoch = 50


    else:
        start_epoch = 0

    '''auxiliary tower'''
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
    else:
        auxiliary_buffer = None
    
    util = int(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
    print('before init generation:', mem)

    if args.gen_images:
        '''init guided gen'''
        from lib.gen.guided_gen import GuidedGen, DeNormalize
        import math
        import torch.distributed as dist
        import torchvision.transforms as transforms

        guided_gen = GuidedGen(dataset=args.dataset)
        if args.dataset in ['cifar10', 'cifar100']:
            denormalize = DeNormalize()
        elif args.dataset == 'imagenet':
            from lib.gen.guided_gen import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            denormalize = DeNormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        tensor2pil = transforms.ToPILImage()

    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
    print('after init generation:', mem)


    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        train_loader.loader.sampler.set_epoch(epoch)
        indices = list(iter(train_loader.loader.sampler))

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs

        # train
        metrics = train_epoch(args, epoch, model, model_ema, train_loader,
                              optimizer, loss_fn, scheduler, auxiliary_buffer,
                              dyrep, loss_scaler, indices, ori_size, steps_per_epoch)

        # validate
        test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
        
        if model_ema is not None:
            test_metrics = validate(args,
                                    epoch,
                                    model_ema.module,
                                    val_loader,
                                    loss_fn,
                                    log_suffix='(EMA)')

        util = int(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
        print('before generation:', mem)


        if args.gen_images:

            if args.dataset in ['cifar10', 'cifar100']:
                gen_interval = 1
                num_images_per_time = 16
                image_size = (32, 32)
            elif args.dataset == 'imagenet':
                gen_interval = 1
                num_images_per_time = int(64 / (dist.get_world_size() / 16))
                image_size = (256, 256)

            if (epoch + 1) % gen_interval == 0 and epoch < int(args.epochs * 0.5):
                guided_gen.rand_gen_ratio = max(1 - epoch / (args.epochs * 0.5 - 1), 0) * 0.5
                # guided_gen.rand_gen_ratio = 1
                _, error_imgs, error_lbls, difficulties = validate(args, epoch, model, gen_val_loader, val_loss_fn, return_error_imgs=True, log_suffix='(Gen val)')
                generated_imgs = []
                error_imgs_pil = []
                bs = 2
                if len(error_imgs) >= num_images_per_time:
                    perm = torch.randperm(len(error_imgs))[:num_images_per_time].cuda()
                else:
                    num_tail_perm = num_images_per_time % len(error_imgs)
                    num_perms = num_images_per_time // len(error_imgs)
                    perm = torch.randperm(len(error_imgs))
                    perm = torch.cat([perm] * num_perms + [perm[:num_tail_perm]], 0).cuda()
                error_imgs = error_imgs[perm]
                error_lbls = error_lbls[perm]
                iters = math.ceil(len(error_imgs) / bs)
                for i in range(iters):
                    logger.info(f"Generating images [{i*bs}/{len(error_imgs)}]")
                    imgs = error_imgs[i*bs:(i+1)*bs].cuda()
                    lbls = error_lbls[i*bs:(i+1)*bs].cuda()
                    gen_imgs = guided_gen(imgs, lbls, model)
                    generated_imgs.extend(gen_imgs)
                    error_imgs_pil.extend([tensor2pil(x) for x in denormalize(imgs).cpu()])
                if args.rank == 0:
                    if not os.path.exists(f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}'):
                        os.makedirs(f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}')
                else:
                    while not os.path.exists(f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}'):
                        time.sleep(1)
                gen_paths = []
                for idx, (img, ori_img, lbl) in enumerate(zip(generated_imgs, error_imgs_pil, error_lbls.tolist())):
                    img.save(f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}/{args.rank}_{idx}_{lbl}.png')
                    ori_img.save(f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}/{args.rank}_{idx}_{lbl}_ori.png')
                    gen_paths.append(f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}/{args.rank}_{idx}_{lbl}.png')

                # extend generated images to dataset
                generated_imgs = [x.resize(image_size) for x in generated_imgs]
                generated_labels = error_lbls[:len(generated_imgs)]

                if args.dataset in ['cifar10', 'cifar100']:
                    extend_cifar_dataset(train_dataset, generated_imgs, generated_labels)
                elif args.dataset == 'imagenet':
                    extend_imagenet_dataset(train_dataset, generated_imgs, generated_labels, path_prefix=f'{FILE_CACHE_PATH}/{args.experiment}/imgs/epoch_{epoch}')
                
                # re-generate sampler
                sampler = train_loader.loader.sampler.__class__(train_dataset, shuffle=train_loader.loader.sampler.shuffle)
                dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
                    pin_memory=False, sampler=sampler, collate_fn=train_loader.loader.collate_fn, drop_last=True, **_LOADER_PARAMS)
                del train_loader.loader._iterator
                del train_loader.loader
                train_loader = DataPrefetcher(dataloader, train_loader.transforms, train_loader.mixup_transform)

            if epoch == int(args.epochs * 0.5):
                if args.dataset in ['cifar10', 'cifar100']:
                    extend_cifar_dataset(train_dataset, gen_val_dataset.data, gen_val_dataset.targets)
                elif args.dataset == 'imagenet':
                    extend_imagenet_dataset(train_dataset, gen_val_dataset.metas)
                # re-generate sampler
                sampler = train_loader.loader.sampler.__class__(train_dataset, shuffle=train_loader.loader.sampler.shuffle)
                dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
                    pin_memory=False, sampler=sampler, collate_fn=train_loader.loader.collate_fn, drop_last=True, **_LOADER_PARAMS)
                del train_loader.loader._iterator
                del train_loader.loader
                train_loader = DataPrefetcher(dataloader, train_loader.transforms, train_loader.mixup_transform)


        # dyrep
        if dyrep is not None:
            if epoch < args.dyrep_max_adjust_epochs:
                if (epoch + 1) % args.dyrep_adjust_interval == 0:
                    # adjust
                    logger.info('DyRep: adjust model.')
                    dyrep.adjust_model()
                    logger.info(
                        f'Model params: {get_params(model)/1e6:.3f} M, FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f} G'
                    )
                    # re-init DDP
                    model = DDP(model.module,
                                device_ids=[args.local_rank],
                                find_unused_parameters=True)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
                elif args.dyrep_recal_bn_every_epoch:
                    logger.info('DyRep: recalibrate BN.')
                    recal_bn(model.module, train_loader, 200)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)

        metrics.update(test_metrics)
        ckpts = ckpt_manager.update(epoch, metrics)
        logger.info('\n'.join(['Checkpoints:'] + [
            '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
        ]))


total_steps = 0

def train_epoch(args,
                epoch,
                model,
                model_ema,
                loader,
                optimizer,
                loss_fn,
                scheduler,
                auxiliary_buffer=None,
                dyrep=None,
                loss_scaler=None,
                indices=None,
                ori_size=0,
                ori_steps_per_epoch=0):
    loss_m = AverageMeter(dist=True)
    data_time_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.train()
    for batch_idx, (input, target) in enumerate(loader):
        batch_indices = indices[batch_idx*len(input):(batch_idx+1)*len(input)]
        batch_indices = torch.LongTensor(batch_indices).cuda()
        data_time = time.time() - start_time
        data_time_m.update(data_time)

        # optimizer.zero_grad()
        # use optimizer.zero_grad(set_to_none=False) for speedup
        for p in model.parameters():
            p.grad = None

        if not args.kd:
            output = model(input)
            loss = loss_fn(output, target)

        else:
            loss = loss_fn(input, target)

        if auxiliary_buffer is not None:
            loss_aux = loss_fn(auxiliary_buffer.output, target)
            loss += loss_aux * auxiliary_buffer.loss_weight

        if loss_scaler is None:
            loss.backward()
        else:
            # amp
            loss_scaler.scale(loss).backward()

        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_max_norm)

        if dyrep is not None:
            # record states of model in dyrep
            dyrep.record_metrics()
            
        if loss_scaler is None:
            optimizer.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        loss_m.update(loss.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Train: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                        'Data: {data_time.val:.2f}s'.format(
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            lr=optimizer.param_groups[0]['lr'],
                            batch_time=batch_time_m,
                            data_time=data_time_m))
        if batch_idx < ori_steps_per_epoch:
            global total_steps
            total_steps += 1
            scheduler.step(total_steps)
        # scheduler.step(epoch * len(loader) + batch_idx + 1)
        # print('log')
        start_time = time.time()

    return {'train_loss': loss_m.avg}


def validate(args, epoch, model, loader, loss_fn, log_suffix='', return_error_imgs=False):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    if return_error_imgs:
        difficulty_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()
    error_imgs = []
    error_labels = []
    difficulties = []
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss.mean()

        if return_error_imgs:
            difficulty = difficulty_criterion(output, target).cpu()
            _, pred = output.topk(1, 1)
            correct = pred.view(-1).eq(target)
            error = ~correct
            error_input = input[error].cpu()
            error_label = target[error].cpu()
            error_imgs.append(error_input)
            error_labels.append(error_label)
            difficulties.append(difficulty[error])

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    if return_error_imgs:
        error_imgs = torch.cat(error_imgs, 0)
        error_labels = torch.cat(error_labels, 0)
        difficulties = torch.cat(difficulties, 0)
        return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}, error_imgs, error_labels, difficulties
    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


if __name__ == '__main__':
    main()
