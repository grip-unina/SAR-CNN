"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.

This is the implementation in Python/Torch of the paper "SAR image despeckling through convolutional neural networks".Results differ slightly from those of the paper, obtained using Matlab/MatConvNet.

"""


import os
import tensorboardX as tbx
import torch

import models.DnCNN as DnCNN
from torch.nn.functional import softplus

class Experiment:
    def __init__(self, basedir, expname=None):
        os.makedirs(basedir, exist_ok=True)

        if expname is None:
            self.expname = utils.get_exp_dir(basedir)
        else:
            self.expname = expname
        self.expdir = os.path.join(basedir, self.expname)

    def create_network(self):
        dncnn_opt = dict(**self.args.dncnn)
        dncnn_opt["residual"] = True
        print(dncnn_opt)
        net = DnCNN.DnCNN(1, 1, **dncnn_opt)
        return net

    def preprocessing_int2net(self, img):
        return img.abs().log()/2

    def postprocessing_net2int(self, img):
        return (2*img).exp()

    def preprocessing_amp2net(self, img):
        return img.abs().log()

    def postprocessing_net2amp(self, img):
        return img.exp()
    
    def preprocessing_log2net(self, img):
        return img

    def create_loss(self):
        def criterion(pred, targets, mask):
            diff = targets - pred  # ==log(R1/R2)/2 in https://www.math.u-bordeaux.fr/~cdeledal/files/articleTIP2009.pdf
            loss = softplus(2.0 * diff) / 2.0 + softplus(-2.0 * diff) / 2.0 - 0.693147180559945  # glrt
            loss = loss.view(pred.shape[0], -1)

            mask = mask.view(pred.shape[0], -1)
            loss = (mask * loss).sum(dim=1)

            return loss
        return criterion

    def create_optimizer(self):
        args = self.args
        assert(args.optimizer == "adam")
        parameters = utils.parameters_by_module(self.net)
        self.base_lr = args.adam["lr"]
        optimizer = torch.optim.Adam(parameters, lr=self.base_lr, weight_decay=args.adam["weightdecay"],
                                     betas=(args.adam["beta1"], args.adam["beta2"]), eps=args.adam["eps"])

        # bias parameters do not get weight decay
        for pg in optimizer.param_groups:
            if pg["name"] == "bias":
                pg["weight_decay"] = 0

        return optimizer

    def learning_rate_decay(self, epoch):
        if epoch < 30:
            return 1
        elif epoch < 50:
            return 0.1
        else:
            return 0

    def setup(self, args=None, use_gpu=True):
        print(self.expname, self.expdir)
        os.makedirs(self.expdir, exist_ok=True)

        if args == None:
            self.args = utils.load_args(self.expdir)
        else:
            self.args = utils.args2obj(args)
            utils.save_args(self.expdir, self.args)

        writer_dir = os.path.join(self.expdir, 'train')
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = tbx.SummaryWriter(log_dir=writer_dir)
        self.use_cuda = torch.cuda.is_available() and use_gpu
        self.net = self.create_network()
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_loss()

        print(self.net)
        print("#Parameter %d" % utils.parameter_count(self.net))

        self.epoch = 0

        if self.use_cuda:
            self.net.cuda()

    def add_summary(self, name, value, epoch=None):
        if epoch is None:
            epoch = self.epoch
        try:
            self.writer.add_scalar(name, value, epoch)
        except:
            pass

def main_sync_sar(args):
    exp_basedir = args.exp_basedir
    patchsize = args.patchsize

    if args.eval:
        from experiment_utility import load_checkpoint, test_list
        from dataset.folders_data import list_test_10synt as listfile_test

        assert(args.exp_name is not None)
        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(use_gpu=args.use_gpu)
        load_checkpoint(experiment, args.eval_epoch)
        outdir = os.path.join(experiment.expdir, "results%03d" % args.eval_epoch)
        test_list(experiment, outdir, listfile_test)
    else:
        from experiment_utility import trainloop
        from dataloaders import create_train_syncsar_dataloaders as create_train_dataloaders
        from dataloaders import create_valid_syncsar_dataloaders as create_valid_dataloaders
        from dataloaders import PreprocessingLogNoisyFromAmp as Preprocessing

        experiment = Experiment(exp_basedir, args.exp_name)
        experiment.setup(args, use_gpu=args.use_gpu)
        trainloader = create_train_dataloaders(patchsize, args.batchsize, args.trainsetiters)
        validloader = create_valid_dataloaders(args.patchsizevalid, args.batchsizevalid)
        trainloop(experiment, trainloader, Preprocessing(), log_data=True, validloader=validloader)

if __name__ == '__main__':
    import argparse
    import os
    from utils import utils
    import torch

    parser = argparse.ArgumentParser(description='SARCNN for SAR image denoising')
    DnCNN.add_commandline_networkparams(parser, "dncnn", 64, 17, 3, "relu", True)

    # Optimizer
    parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd"]) # which optimizer to use
    # parameters for Adam
    parser.add_argument("--adam.beta1", type=float, default=0.9)
    parser.add_argument("--adam.beta2", type=float, default=0.999)
    parser.add_argument("--adam.eps", type=float, default=1e-8)
    parser.add_argument("--adam.weightdecay", type=float, default=1e-4)
    parser.add_argument('--adam.lr', type=float, default=0.001)
    # parameters for SGD
    parser.add_argument("--sgd.momentum", type=float, default=0.9)
    parser.add_argument("--sgd.weightdecay", type=float, default=1e-4)
    parser.add_argument('--sgd.lr', type=float, default=0.001)

    # Eval mode
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_epoch', type=int)

    # Training options
    parser.add_argument("--batchsize"     , type=int, default=128)
    parser.add_argument("--patchsize"     , type=int, default=40 )
    parser.add_argument("--batchsizevalid", type=int, default=8  )
    parser.add_argument("--patchsizevalid", type=int, default=256)

    # Misc
    utils.add_commandline_flag(parser, "--use_gpu", "--use_cpu", True)
    parser.add_argument("--exp_name"   , default=None)

    base_expdir = "./results/sar_sync/sarcnn17/"
    parser.add_argument("--exp_basedir", default=base_expdir)
    parser.add_argument("--trainsetiters", type=int, default=640)
    args = parser.parse_args()
    main_sync_sar(args)
