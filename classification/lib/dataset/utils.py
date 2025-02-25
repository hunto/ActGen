import torch
import copy
import torch.distributed as dist
import numpy as np
from PIL import Image
import os


def split_cifar_dataset(dataset, portion=0.5, shuffle=True):
    dataset2 = copy.deepcopy(dataset)
    size1 = int(len(dataset) * portion)
    size2 = len(dataset) - size1
    if shuffle:
        perm = torch.randperm(len(dataset)).cuda()
        if dist.is_initialized():
            dist.broadcast(perm, 0)
        perm = perm.cpu().numpy()
    else:
        perm = torch.arange(len(dataset)).numpy()

    dataset2.data = dataset.data[perm[size1:]]
    dataset2.targets = np.array(dataset.targets)[perm[size1:]]
    dataset.data = dataset.data[perm[:size1]]
    dataset.targets = np.array(dataset.targets)[perm[:size1]]
    return dataset, dataset2

def split_imagenet_dataset(dataset, portion=0.5, shuffle=True):
    dataset2 = copy.deepcopy(dataset)
    size1 = int(len(dataset) * portion)
    size2 = len(dataset) - size1
    if shuffle:
        perm = torch.randperm(len(dataset)).cuda()
        if dist.is_initialized():
            dist.broadcast(perm, 0)
        perm = perm.cpu().numpy()
    else:
        perm = torch.arange(len(dataset)).numpy()
    
    dataset2.metas = np.array(dataset.metas, dtype=object)[perm[size1:]]
    dataset.metas = np.array(dataset.metas, dtype=object)[perm[:size1]]
    return dataset, dataset2


def extend_cifar_dataset(dataset, images, labels):
    labels = np.array(labels)
    if not isinstance(images, np.ndarray):
        # convert PIL images to numpy
        data = np.stack([np.array(x) for x in images])  # NHWC

        # gather all images from all ranks
        tensor_data = torch.tensor(data).cuda()
        all_data = [torch.empty_like(tensor_data, device=tensor_data.device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_data, tensor_data)
        data = torch.cat(all_data, 0).cpu().numpy()
        labels = torch.from_numpy(labels).cuda()
        all_labels = [torch.empty_like(labels, device=labels.device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_labels, labels)
        labels = torch.cat(all_labels, 0).cpu().numpy()
    else:
        data = images

    dataset.data = np.concatenate([dataset.data, data], 0)

    if isinstance(dataset.targets, (np.ndarray)):
        dataset.targets = np.concatenate([dataset.targets, labels], 0)
    else:
        dataset.targets.extend(labels.tolist())


def extend_imagenet_dataset(dataset, images_or_metas, labels=None, path_prefix=''):
    # if not isinstance(images_or_metas[0][0], str):
    if isinstance(images_or_metas[0], Image.Image):
        labels = np.array(labels)

        # gather images and labels
        # images = np.stack([np.array(img) for img in images_or_metas])
        # tensor_data = torch.tensor(images).cuda()
        # all_data = [torch.empty_like(tensor_data, device=tensor_data.device) for _ in range(dist.get_world_size())]
        # dist.all_gather(all_data, tensor_data)
        # images = torch.cat(all_data, 0).cpu().numpy()
        # images = [Image.fromarray(img) for img in images]

        labels = torch.from_numpy(labels).cuda()
        all_labels = [torch.empty_like(labels, device=labels.device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_labels, labels)
        labels = torch.cat(all_labels, 0).cpu().numpy()

        images = []
        lbl_idx = 0
        for rank in range(dist.get_world_size()):
            for idx in range(len(images_or_metas)):
                lbl = labels[lbl_idx]
                img_name = os.path.join(path_prefix, f'{rank}_{idx}_{lbl}.png')
                images.append('<gen>' + img_name)
                lbl_idx += 1

        new_data = []
        for image, label in zip(images, labels):
            new_data.append([image, label])
        new_data = np.array(new_data, dtype=object)
    else:
        new_data = images_or_metas
    if isinstance(dataset.metas, np.ndarray):
        dataset.metas = np.concatenate([dataset.metas, new_data], 0)
    else:
        dataset.metas.extend(new_data.tolist())
    dataset._mc_initialized = False
    if hasattr(dataset, 'backend'):
        del dataset.backend