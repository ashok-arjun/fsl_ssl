# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
import math

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset, Sampler


# Below: just for safety, since DDP can be used. Should be removed later

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)


def get_patches(img, transform_jigsaw, transform_patch_jigsaw, permutations):
    if np.random.rand() < 0.30:
        img = img.convert('LA').convert('RGB')## this should be L instead....... need to change that!!

    img = transform_jigsaw(img)

    s = float(img.size[0]) / 3
    a = s / 2
    tiles = [None] * 9
    for n in range(9):
        i = int(n / 3)
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a ), int(c[0] + a )]).astype(int)
        tile = img.crop(c.tolist())
        tile = transform_patch_jigsaw(tile)
        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        tiles[n] = tile
        
    order = np.random.randint(len(permutations))
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data, 0)

    return data, int(order)

def retrive_permutations(classes):
    all_perm = np.load('permutations_%d.npy' % (classes))
    if all_perm.min() == 1:
        all_perm = all_perm - 1

    return all_perm

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity, \
                jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None, \
                rotation=False, isAircraft=False, grey=False, return_name=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        self.permutations = retrive_permutations(35)

        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey
        self.return_name = return_name

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        
        if self.grey:
            img = Image.open(image_path).convert('L').convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        
        if self.isAircraft:
            ## crop the banner
            img = img.crop((0,0,img.size[0],img.size[1]-20))
        
        if self.jigsaw:
            patches, order = get_patches(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations)
        if self.rotation:
            rotated_imgs = [
                    self.transform(img),
                    self.transform(img.rotate(90,expand=True)),
                    self.transform(img.rotate(180,expand=True)),
                    self.transform(img.rotate(270,expand=True))
                ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
        
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])

        if self.jigsaw:
            if self.return_name:
                return img, target, patches, order, image_path
            else:
                return img, target, patches, order
        elif self.rotation:
            if self.return_name:
                return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels, image_path
            else:
                return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
        else:
            if self.return_name:
                return img, target, image_path
            else:
                return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform, jigsaw=False, \
                transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, isAircraft=False, grey=False):
        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey

        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform, jigsaw=self.jigsaw, \
                                    transform_jigsaw=self.transform_jigsaw, transform_patch_jigsaw=self.transform_patch_jigsaw, \
                                    rotation=self.rotation, isAircraft=self.isAircraft, grey=self.grey)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, \
                jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, isAircraft=False, grey=False):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey

        self.jigsaw = jigsaw
        if jigsaw:
            self.permutations = retrive_permutations(35)
            self.transform_jigsaw = transform_jigsaw
            self.transform_patch_jigsaw = transform_patch_jigsaw

    def __getitem__(self,i):
        image_path = os.path.join(self.sub_meta[i])
        if self.grey:
            img = Image.open(image_path).convert('L').convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')

        if self.isAircraft:
            ## crop the banner
            img = img.crop((0,0,img.size[0],img.size[1]-20))

        if self.jigsaw:
            patches, order = get_patches(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations)
        if self.rotation:
            rotated_imgs = [
                    self.transform(img),
                    self.transform(img.rotate(90,expand=True)),
                    self.transform(img.rotate(180,expand=True)),
                    self.transform(img.rotate(270,expand=True))
                ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
        img = self.transform(img)
        target = self.target_transform(self.cl)
        
        if self.jigsaw:
            return img, target, patches, order
        elif self.rotation:
            return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
        else:
            return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

            
class DatasetFromSampler(Dataset):
    
    # Taken from https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/sampler.py#L522
    
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)            


class DistributedSamplerWrapper(DistributedSampler):
    
    # Modified from https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/sampler.py#L522
    
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))