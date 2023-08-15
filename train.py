import numpy as np

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

from trainer import train_one_epoch
from functools import partial
import os

   
   

name = "BASE_DRIVE_SEG_01"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.system(f"rm -rf runs/bseg_{name}")
writer = SummaryWriter('runs/bseg_{}'.format(name))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_number = 0
EPOCHS = 400
LR=0.0002
best_vloss = np.Inf
input_size = 512

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

tr_dataloader = data["tr"]["dataloader"]
vl_dataloader = data["vl"]["dataloader"]
# --------------------------------------------------------------



loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCEWithLogitsLoss().to(device)
# loss_fn = nn.BCELoss().to(device)
# def dice_loss(y_true, y_pred, smooth=1e-5):
#     intersection = torch.sum(y_true * y_pred, dim=(1,2,3))
#     sum_of_squares_pred = torch.sum(torch.square(y_pred), dim=(1,2,3))
#     sum_of_squares_true = torch.sum(torch.square(y_true), dim=(1,2,3))
#     dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
#     return dice.mean()
# loss_fn = partial(dice_loss, smooth=1e-5)


# --------------------------------------------------
model = TSegDiff(
    input_hw=(input_size, input_size),
    in_ch=3,
    out_ch=1,
    init_filter=64,
    patch_size=(16, 16),
    latent_dim=input_size,
)

# model = UNet(
#     out_channels=1, 
#     in_channels=3, 
#     depth=5,
#     start_filts=64, 
#     up_mode='transpose', 
#     merge_mode='concat'
# )

# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=False)


model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, last_epoch=epochs, verbose=True)
# step_lr_schedule = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# -------------------------------------------------------




for epoch in range(EPOCHS):

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, model, optimizer, loss_fn, tr_dataloader, writer, device)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(vl_dataloader):
            vimgs, vmsks = vdata["image"].to(device), vdata["mask"].to(device)

            voutputs = model(vimgs)
            vloss = loss_fn(voutputs, vmsks)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'ep {epoch_number+1:03d}/{EPOCHS:03d}:-> loss-tr: {avg_loss:.6f}, loss-vl: {avg_vloss:.6f}')

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = best_model_path
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
    
