import numpy as np
import json
import torch
from torch import nn
from torch import optim

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from tqdm import tqdm
from dsl.datasets import get_drive

from models.trans import MTv00
from models.unet import UNet
from models.tseg import *

from metrics import get_binary_metrics
from utils import write_imgs

   
   

name = "BASE_DRIVE_SEG_01"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/bseg_{name}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 1024
patch_size=(16, 16)
init_filter=128

best_model_path = f'/cabinet/afshin/base_seg/saved_model/model_{name}'


# --------------------------------------------------------------
data_dir = "/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/dataset/Retinal/DRIVE"
data_dir = "/cabinet/dataset/Retinal/DRIVE"
data = get_drive(loading="fast",                
                 data_dir=data_dir,
                 size=(input_size, input_size),
                 tr_te_p=0.7,
                 tr_vl_p=0.75,
                 one_hot=False,
                 tr_dataloader_args={"batch_size": 4, "shuffle": True},
                 vl_dataloader_args={"batch_size": 4, "shuffle": True},
                 te_dataloader_args={"batch_size": 4, "shuffle": True},
                 force_prepare=False,
                 logger=None)

te_dataloader = data["te"]["dataloader"]
# --------------------------------------------------------------




# --------------------------------------------------
# model = MTv00(
#     in_ch=3,
#     out_ch=1
# )

model = TSegDiff(
    input_hw=(input_size, input_size),
    in_ch=3,
    out_ch=1,
    init_filter=init_filter,
    patch_size=patch_size,
    latent_dim=input_size,
)

# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=True)

checkpoint = torch.load(best_model_path, map_location="cpu")
model.load_state_dict(checkpoint)
        
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {total_params}")
# -------------------------------------------------------


test_metrics = get_binary_metrics()



model.eval()

# Disable gradient computation and reduce memory consumption.
with torch.no_grad():
    for i, data in tqdm(enumerate(te_dataloader)):
        imgs, msks = data["image"].to(device), data["mask"].to(device)

        prds = model(imgs)
        
        if msks.shape[1] > 1:
            prds_ = torch.argmax(prds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
        else:
            prds_ = torch.where(prds > 0, 1, 0).float()
            msks_ = torch.where(msks > 0, 1, 0)
        
        test_metrics.update(prds_, msks_)
        
        write_imgs(writer, imgs, msks, prds, i, name, dataset="drive".upper())

        
result = test_metrics.compute()
writer.add_scalars(
    f"Metrics/test-{name}",
    result,
)

print(json.dumps({k: v.item() for k, v in result.items()}, indent=4))

# writer.flush()


    
