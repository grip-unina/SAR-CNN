from scipy.io import loadmat, savemat
from PIL import Image
from os import makedirs
import numpy as np
seed = np.random.RandomState(112311)

makedirs('./sets/Noisy100')
for indexN in range(10):
    for indexI in range(1,11):
        img = np.array(Image.open('./sets/Set12/%02d.png'%indexI))
        
        dat = dict()
        dat['norm_int'] = 65536.0
        dat['target_int'] = ((np.float32(img)+1.0)/256.0)**2
        dat['mask'] = np.ones(dat['target_int'].shape, np.float32)
        dat['noisy_int'] = dat['target_int'] * seed.gamma(size=dat['target_int'].shape, shape=1.0, scale=1.0).astype(dat['target_int'].dtype)

        filename_out = './sets/Noisy100/synt%02d_%02d.mat'%(indexI,indexN)
        
        print(indexI, indexN, filename_out)
        savemat(filename_out, dat)

