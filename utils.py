from matplotlib import pyplot as plt
import numpy as np
import cv2
import yaml
from termcolor import colored
import os
import json
import torch
from scipy import ndimage
from skimage import feature
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


jet = plt.get_cmap("jet")


def write_imgs(
    writer,
    imgs, msks, prds, 
    step, id, dataset, 
    ids=None
):
    if imgs.shape[1] == 1: 
        imgs = torch.stack(3*[imgs[:,0],],1)
    
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    img_grid = torchvision.utils.make_grid(imgs)
    msk_grid = torchvision.utils.make_grid(msks)
    prd_grid = torchvision.utils.make_grid(prds)

    prds_jet = torch.zeros_like(imgs)
    for i, prd in enumerate(prds.detach().cpu().numpy()):
        t = jet(prd[0]).transpose(2, 0, 1)[:-1, :, :]
        t = np.log(t + 0.1)
        t = (t - t.min()) / (t.max() - t.min())
        prds_jet[i, :, :, :] = torch.tensor(t)

    prds_jet_grid = torchvision.utils.make_grid(prds_jet)
    img_msk_prd_grid = torch.concat(
        [
            img_grid,
            torch.stack(3*[msk_grid[0],],0),
            torch.stack(3*[prd_grid[0]>0,],0),
            torch.stack(3*[prds_jet_grid[0],], 0),
        ],
        dim=1,
    )
    # writer.add_image(f'ISIC 2018 - Results/{"-".join(ids)}', img_msk_prd_grid)
    writer.add_image(f"{dataset}/Test:{id}", img_msk_prd_grid, step)

    
    
    
    


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class _bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def cprint(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{_bcolors.ENDC}"

    if "bold" in p.lower():
        pre += _bcolors.BOLD
    elif "underline" in p.lower():
        pre += _bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += _bcolors.HEADER

    if "warning" in p.lower():
        pre += _bcolors.WARNING
    elif "error" in p.lower():
        pre += _bcolors.FAIL
    elif "ok" in p.lower():
        pre += _bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += _bcolors.OKBLUE
        else:
            pre += _bcolors.OKCYAN

    print(f"{pre}{string}{_bcolors.ENDC}")


_print = cprint


def load_config(config_filepath):
    try:
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)



def print_config(config, logger=None):
    conf_str = json.dumps(config, indent=2)
    if logger:
        logger.info(f"\n{' Config '.join(2*[10*'>>',])}\n{conf_str}\n{28*'>>'}")
    else:
        _print("Config:", "info_underline")
        print(conf_str)
        print(30 * "~-", "\n")
        



def standardize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(tensor, dim=(-1, -2), keepdim=True)
    std = torch.std(tensor, dim=(-1, -2), keepdim=True)
    return (tensor - mean) / std



from copy import deepcopy
def draw_boundary(x, img, color):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = (img-img.min())/(img.max()-img.min())

    edged = calc_edge(x[0,:], mode='canny')
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for idx, imc in enumerate(range(3)):
        img[idx,:,:] = cv2.drawContours(np.uint8(img[idx,:,:]*255), contours, -1, color[idx], 1)/255.
    return img
