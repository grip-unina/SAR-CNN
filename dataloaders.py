"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
 
"""

import torch
from dataset import folders_data
from dataset import sar_dataset
from torchvision.transforms import Compose

scale_img = 255.0

def create_valid_awgn_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropPil(patchsize),
        sar_dataset.PilToGrayTensor(bayes=0.0,scale=scale_img),
    ])

    validset = sar_dataset.PlainImageFolder(dirs=folders_data.valid68_dir, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=1)

    return validloader

def create_valid_syncsar_dataloaders(patchsize, batchsize):
    print('valid_syncsar:', batchsize, patchsize)
    transform_valid = Compose([
        sar_dataset.CenterCropPil(patchsize),
        sar_dataset.PilToGrayTensor(bayes=1.0,scale=scale_img),
    ])

    validset = sar_dataset.PlainImageFolder(dirs=folders_data.valid68_dir, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=1)

    return validloader

def create_valid_realsar_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropNy(patchsize),
        sar_dataset.NumpyToTensor(),
    ])

    validset = sar_dataset.PlainSarFolder(dirs=folders_data.valid_mlook_dir, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=1)

    return validloader

def create_train_awgn_dataloaders(patchsize, batchsize, trainsetiters):
    transform_train = Compose([
        sar_dataset.RandomCropPil(patchsize),
        sar_dataset.Random8OrientationPil(),
        sar_dataset.PilToGrayTensor(bayes=0.0, scale=scale_img),
    ])

    trainset = sar_dataset.PlainImageFolder(dirs=folders_data.train400_dir, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=20)

    return trainloader

def create_train_syncsar_dataloaders_old(patchsize, batchsize, trainsetiters):
    import torchvision.transforms as transforms
    transform_train = Compose([
        transforms.RandomCrop(patchsize),
        sar_dataset.RandomOrientation90Pil(),
        #sar_dataset.Random8OrientationPil(),
        transforms.RandomVerticalFlip(),
        sar_dataset.ToGrayscale(),
        transforms.ToTensor(),
        sar_dataset.AddBayes(),
    ])

    train_folders = folders_data.train400_dir

    trainset = sar_dataset.PlainImageFolder(dirs=train_folders, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset] * trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=20)

    return trainloader


def create_train_syncsar_dataloaders(patchsize, batchsize, trainsetiters):
    print('train_syncsar:', trainsetiters, batchsize, patchsize)
    transform_train = Compose([
        sar_dataset.RandomCropPil(patchsize),
        sar_dataset.Random8OrientationPil(),
        sar_dataset.PilToGrayTensor(bayes=1.0,scale=scale_img),
    ])

    trainset = sar_dataset.PlainImageFolder(dirs=folders_data.train400_dir, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset] * trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=20)

    return trainloader

def create_train_realsar_dataloaders(patchsize, batchsize, trainsetiters):
    transform_train = Compose([
        sar_dataset.RandomCropNy(patchsize),
        sar_dataset.Random8OrientationNy(),
        sar_dataset.NumpyToTensor(),
    ])

    trainset = sar_dataset.PlainSarFolder(dirs=folders_data.train_mlook_dir, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=20)

    return trainloader

class PreprocessingRealInt:
    def __call__(self, inputs):
        noisy  = inputs[:, 0:1, :, :]
        target = inputs[:, 1:2, :, :]
        mask   = inputs[:, 2:3, :, :]

        return noisy, target, mask

class PreprocessingLogNoisyFromAmp:
    def __init__(self, flag_bayes=False):
        from torch.distributions.gamma import Gamma
        self.gen_dist = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        self.flag_bayes = flag_bayes
    def __call__(self, target):
        if self.flag_bayes:
            target = target + 1 / scale_img
        target = target.log()
        noise = self.gen_dist.sample(target.shape)[:, :, :, :, 0]
        noise = noise.log() / 2
        mask  = torch.ones(target.shape)
        if target.is_cuda:
            noise = noise.cuda()
            mask  = mask.cuda()
        noisy = target + noise
        return noisy, target, mask

class PreprocessingIntNoisyFromAmp:
    def __init__(self, flag_bayes=False):
        from torch.distributions.gamma import Gamma
        self.gen_dist = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        self.flag_bayes = flag_bayes
    def __call__(self, target):
        if self.flag_bayes:
            target = target + 1 / scale_img
        target = target ** 2
        noise = self.gen_dist.sample(target.shape)[:, :, :, :, 0]
        mask  = torch.ones(target.shape)
        if target.is_cuda:
            noise = noise.cuda()
            mask  = mask.cuda()
        noisy = target * noise
        return noisy, target, mask

if __name__ == '__main__':
    #data_iterator = create_valid_realsar_dataloaders(256, 8)
    #data_iterator = create_train_realsar_dataloaders(256, 32, 1)
    #data_preprocessing = PreprocessingRealInt(); flag_log = False

    data_iterator = create_valid_syncsar_dataloaders(256, 16)
    #data_iterator = create_train_syncsar_dataloaders(256, 100, 1)
    #data_preprocessing = PreprocessingLogNoisyFromAmp(); flag_log = True
    data_preprocessing = PreprocessingIntNoisyFromAmp(); flag_log = False

    import matplotlib.pyplot as plt
    for index, patch in enumerate(data_iterator):
        noisy, target, mask = data_preprocessing(patch)
        if flag_log:
            noisy = noisy.exp()
            target = target.exp()
        else:
            noisy = noisy.sqrt()
            target = target.sqrt()
        print(index, patch.shape, noisy.shape)
        plt.figure()
        plt.subplot(1,3,1); plt.imshow(noisy[0,0] , clim=[0, 1], cmap='gray')
        plt.subplot(1,3,2); plt.imshow(target[0,0], clim=[0, 1],cmap='gray')
        plt.subplot(1,3,3); plt.imshow(mask[0,0]  , clim=[0, 1], cmap='gray')
        plt.show()

