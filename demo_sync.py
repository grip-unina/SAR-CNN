"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
 
"""

image_id = 7
noise_seed = 32
use_cuda = True

import os
import torch
from PIL import Image
import numpy as np
from utils import metrics
import pickle


# network SAR-CNN 2017
import models.DnCNN as DnCNN
with open("./weights/sar_sync/SAR_CNN_e50.pkl", "rb") as fid:
    dncnn_opt = dict(**pickle.load(fid).dncnn)
    dncnn_opt["residual"] = True
net = DnCNN.DnCNN(1, 1, **dncnn_opt)
net.load_state_dict(torch.load('./weights/sar_sync/SAR_CNN_e50.t7')['net'])
pad = 0
    
def preprocessing_int2net(img):
    return img.abs().log()/2

def postprocessing_net2int(img):
    return (2*img).exp()
    

target_amp   = (np.float32(np.array(Image.open('./sets/Set12/%02d.png'%image_id))) + 1.0)/256.0
randomStream = np.random.RandomState(noise_seed)
noise_int    = randomStream.gamma(size=target_amp.shape, shape=1.0, scale=1.0).astype(target_amp.dtype)

with torch.no_grad():
    if use_cuda:
        net = net.cuda()
        
    target_amp = torch.from_numpy(target_amp)[None, None, :, :]
    noise_int  = torch.from_numpy(noise_int)[None, None, :, :]
    noisy_int  = (target_amp**2)*noise_int
    if pad>0:
        noisy_pad = torch.nn.functional.pad(noisy_int, (pad, pad, pad, pad), mode='reflect', value=0)
    else:
        noisy_pad = noisy_int
        
    noisy_pad =  preprocessing_int2net(noisy_pad)
    if use_cuda:
        pred_int = net(noisy_pad.cuda()).cpu()
    else:
        pred_int = net(noisy_pad)
    pred_int = postprocessing_net2int(pred_int)
    
    noisy_amp = noisy_int.abs().sqrt()
    pred_amp  = pred_int.abs().sqrt()
    stats_one = dict()
    stats_one["mse"]  = metrics.metric_mse(pred_amp, target_amp, size_average=True).data
    stats_one["psnr"] = metrics.metric_psnr(pred_amp, target_amp, maxval=1.0, size_average=True).data
    stats_one["ssim"] = metrics.metric_ssim(pred_amp, target_amp, size_average=True).data

    target_amp = target_amp[0,0,:,:].numpy()
    noisy_amp  = noisy_amp[0,0,::].numpy()
    pred_amp   = pred_amp[0,0,::].numpy()
    
images = np.hstack((target_amp,noisy_amp,pred_amp))
images = (256*images-1.0).clip(0,255).astype(np.uint8)
Image.fromarray(images).save('result.png')
    
print(stats_one)
            

    
        