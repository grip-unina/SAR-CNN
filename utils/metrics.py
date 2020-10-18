import numpy as np
import torch

def n10log10(x):
    return -10.0*x.log() / float(np.log(10))

def metrics_mean(metric, keepdim=False):
    return metric.view(metric.shape[0],-1).mean(dim=1, keepdim=keepdim)

def metrics_mask(metric, mask, keepdim=False):
    metric = (mask*metric).view(metric.shape[0],-1).sum(dim=1, keepdim=keepdim)
    mask = mask.view(metric.shape[0],-1).sum(dim=1, keepdim=keepdim)
    mask = torch.clamp(mask, min=0.001)
    
    return metric/mask

def metric_mse(x, y, size_average=True):
    mse = ((x-y)**2)
    mse = metrics_mean(mse)

    if size_average:
        mse = mse.mean()

    return mse

def metric_psnr(x,y, maxval=1, size_average=True):
    mse = metric_mse(x, y, size_average=False)
    psnr = n10log10(mse/(maxval**2))

    if size_average:
        psnr = psnr.mean()

    return psnr

def metric_mse_mask(x, y, mask, size_average=True):
    mse = ((x - y) ** 2)
    mse = metrics_mask(mse, mask)

    if size_average:
        mse = mse.mean()

    return mse

def metric_psnr_mask(x, y, mask, maxval=1, size_average=True):
    mse  = metric_mse_mask(x, y, mask, size_average=False)
    psnr = n10log10(mse/(maxval**2))

    if size_average:
        psnr = psnr.mean()
    return psnr

# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_map(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def metric_ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_map = _ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def metric_ssim_mask(img1, img2, mask, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_map = _ssim_map(img1, img2, window, window_size, channel)
    ssim = metrics_mask(ssim_map, mask)

    if size_average:
        ssim = ssim.mean()
    return ssim


    