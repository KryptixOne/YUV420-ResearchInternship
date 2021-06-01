# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import *
from compressai.models import CompressionModel, ScaleHyperpriorYUV_SEP,ScaleHyperpriorYUV_Shuffle
from compressai.models.utils import conv, deconv
from compressai.layers import GDN
from compressai.transforms.functional import yuv_444_to_420, yuv_420_to_444
from torchsummary import summary

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def test_epoch_with_save(epoch, test_dataloader, model, criterion,TrainingType,LL,deviceIn,finalName):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    stringLambdaValue = str(LL)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    if TrainingType == 1:
        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                d_changed,paddedDim,paddedDim0Flag,paddedDim1Flag=PadrForNonSquare(d,TrainingType,deviceIn)
                out_net = model(d_changed)
                out_criterion = criterion(out_net, d_changed)
                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])

                imgY_new, imgU_new, imgV_new = DepadTensor(out_net["x_hat"], paddedDim, paddedDim0Flag, paddedDim1Flag,TrainingType)
                filename = finalName
                bit_flag ='uint8'

                imWrite_YUV(imgY_new,imgU_new,imgV_new,filename,bit_flag)



    elif TrainingType ==2:
        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                d_changed, paddedDim, paddedDim0Flag, paddedDim1Flag = PadrForNonSquare(d,TrainingType,deviceIn)
                out_net = model(d_changed)
                out_criterion = criterion(out_net, d_changed)
                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                imgY_new, imgU_new, imgV_new = DepadTensor(out_net["x_hat"], paddedDim, paddedDim0Flag, paddedDim1Flag,TrainingType)
                filename = finalName
                bit_flag = 'uint8'
                imWrite_YUV(imgY_new, imgU_new, imgV_new, filename, bit_flag)




    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Quality Test script.")
    ##    parser.add_argument(
    ##        "-m",
    ##        "--model",
    ##        default="bmshj2018-factorized",
    ##        choices=models.keys(),
    ##        help="Model architecture (default: %(default)s)",
    ##    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=12,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(128, 128),  # needs to be factorable by 2
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    #checkpoint
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint according to appropriate lambda and training type")

    parser.add_argument("--FinalFileName",
                        type=str,
                        help="Path name to Reconstructed Image. Should be full pathname + basename")
    parser.add_argument(
        "--training_type",
        default=1,
        type=int,
        help="1 = Separate Branches, 2 = PixelShuffle (default: %(default)s",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    #Listed to allow for multiple Lambdas to be used for Reconstruction
    lambdalist = []
    lambdalist.append((args.lmbda))


    for L in lambdalist:
        LambdaStr = str(L)
        checkpointstr =args.checkpoint
        if args.seed is not None:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
        TrainingType = args.training_type
        test_dataset = YUVImageFolder(args.dataset, split="test",Training_Type=TrainingType)

        device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        print('running with', device)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

        # print('after data')
        if TrainingType == 1:
            net = ScaleHyperpriorYUV_SEP(128, 192)
        elif TrainingType == 2:
            net = ScaleHyperpriorYUV_Shuffle(128, 192)


        net = net.double()
        net = net.to(device)
        #summary(net, (3, 256, 256), 8) # if you wish to see the summary, change net.double() to net.float()

        optimizer, aux_optimizer = configure_optimizers(net, args)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        print('Using Lambda = ', str(L))
        criterion = RateDistortionLoss(lmbda=L)

        last_epoch = 0
        if checkpointstr:  # load from previous checkpoint
            print("Loading", checkpointstr)
            checkpoint = torch.load(checkpointstr, map_location=device)
            last_epoch = checkpoint["epoch"] + 1
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        finalName= args.FinalFileName
        loss = test_epoch_with_save(1, test_dataloader, net, criterion,TrainingType,L,device,finalName)

if __name__ == "__main__":
    main(sys.argv[1:])
