"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
 
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def sar_filter(filename):
    ext = os.path.splitext(filename)[1]
    return (ext=='.mat') or (ext=='.hdf5')

def image_filter(filename):
    from torchvision.datasets.folder import IMG_EXTENSIONS
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def sar_loader(path):
    import h5py
    with h5py.File(path, 'r') as fid:
        data = np.copy(fid['data'][()])
    assert(data.shape[0]==3)
    return data

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
    return img

class ToGrayscale(object):
    def __call__(self, img):
        return img.convert('L', (0.2989, 0.5870, 0.1140, 0))

class AddBayes(object):
    def __call__(self, img):
        return img + 1.0/255.0

class PilToGrayTensor(object):
    def __init__(self, bayes=0.0, scale = 256.0):
        self.bayes = bayes
        self.scale = scale
    def __call__(self, pic):
        if (pic.mode!='L'):
            pic = pic.convert('L', (0.2989, 0.5870, 0.1140, 0))
        assert(pic.mode=='L')
        nchannel = 1
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().add(self.bayes).div(self.scale)
    def __repr__(self):
        return self.__class__.__name__ + '(bayes={0},scale={1})'.format(self.bayes,self.scale)

class Amp2Int(object):
    def __call__(self, x):
        return x**2

class NumpyToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x.copy(order='C'))

class CenterCropPil(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        y1 = (img.size[1] - self.size)//2
        x1 = (img.size[0] - self.size)//2
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img.crop((x1,y1,x2,y2)) #(left, upper, right, lower)-tuple.
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
class RandomCropPil(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        y1 = np.random.randint(0, img.size[1] - self.size)
        x1 = np.random.randint(0, img.size[0] - self.size)
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img.crop((x1,y1,x2,y2)) #(left, upper, right, lower)-tuple.
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
class Random8OrientationPil(object):
    def __call__(self, img):
        PIL_FLIPS = (None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270,
                     Image.FLIP_LEFT_RIGHT, Image.TRANSPOSE, Image.FLIP_TOP_BOTTOM, Image.TRANSVERSE)
        ind = np.random.randint(0,8)
        if ind>0:
            img_out = img.transpose(PIL_FLIPS[ind])
        else:
            img_out = img.copy()
        return img_out

class RandomOrientation90Pil(object):
    def __call__(self, img):
        import numpy as np
        degrees = 90*np.random.randint(0,4)
        img.rotate(degrees)
        return img

class RandomOrientation90(object):
    def __call__(self, img):
        import numpy as np
        degrees = 90*np.random.randint(0,4)
        img.rotate(degrees)
        return img



class CenterCropNy(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        y1 = (img.shape[1] - self.size)//2
        x1 = (img.shape[2] - self.size)//2
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img[:, y1:y2, x1:x2]
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
class RandomCropNy(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        y1 = np.random.randint(0, img.shape[1] - self.size)
        x1 = np.random.randint(0, img.shape[2] - self.size)
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img[:, y1:y2, x1:x2]
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
class Random8OrientationNy(object):
    def __init__(self):
        pass
    def __call__(self, img):
        k = img.shape[0]
        rot = np.random.randint(0,8)
        img = np.rot90(img, axes=(1,2), k=rot)
        if rot>3:
            img = img[:,::-1,:]
        assert(img.shape[0]==k)
        return img

def find_files(dir, filter=None):
    images = list()
    if filter is None:
        filter = lambda x: True

    for fname in sorted(os.listdir(dir)):
        if filter(fname):
            images.append(os.path.join(dir, fname))

    return images

class PlainImageFolder(Dataset):
    r"""
    Adapted from torchvision.datasets.folder.ImageFolder
    """

    def __init__(self, dirs, transform=None, cache=False, loader = pil_loader, filter=image_filter):
        self.cache = cache
        self.img_cache = {}
        if isinstance(dirs, list):
            imgs = list()
            for r in dirs:
                imgs.extend(find_files(r, filter=filter))
        else:
            imgs = find_files(dirs, filter=filter)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + dirs ))

        self.dirs = dirs
        self.imgs = imgs
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        path = self.imgs[index]
        if not index in self.img_cache:
            img = self.loader(path)
            if self.cache:
                self.img_cache[index] = img
        else:
            img = self.img_cache[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)

class PlainSarFolder(PlainImageFolder):
    def __init__(self, dirs, transform=None, cache=False):
        PlainImageFolder.__init__(self, dirs, transform=transform, cache=cache, loader = sar_loader, filter=sar_filter)

if __name__=="__main__":
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    file = "/nas184/experiments/davide.cozzolino/otherMethos/visinf/n3net/datasets/sets/train400/16004.jpg"
    img = Image.open(file).convert('L', (0.2989, 0.5870, 0.1140, 0))
    t = to_tensor(img)



