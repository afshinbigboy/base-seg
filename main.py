import numpy as np

import torch
from torch import nn
from torch import optim

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from losses import BinaryDiceLoss

from tqdm import tqdm
from dsl.datasets import get_drive

from models.trans import MTv00
from models.unet import UNet
from models.tseg import *
# from models.daeformer import DAEFormer
# from other_models import MERIT_Cascaded


from trainer import train_one_epoch
from functools import partial
import os, sys, json

from metrics import get_binary_metrics
from utils import write_imgs, load_config, print_config
from augs import DataAugmentationTransform, A

import signal   
    

kill = 0
def sighandler(*args):
    global kill
    print("OKKKKKKKK")
    kill=True
signal.signal(signal.SIGINT, sighandler)
    
    
    
   

name = "BASE_DRIVE_SEG_01"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.system(f"rm -rf runs/bseg_{name}")
writer = SummaryWriter('runs/bseg_{}'.format(name))


epoch_number = 0
EPOCHS = 2000
LR=0.0001
best_vloss = np.Inf

# ----- 0.65 ------ init k=3
# input_size = 512
# patch_size=(8, 8)
# init_filter=128   
# latent_dim=1024

# ----- 0.71 ------ init k=5
# input_size = 512
# patch_size=(8, 8)
# init_filter=64
# latent_dim=512

# ----- 0.726 ------ init k=5
# input_size = 512
# patch_size=(4, 4)
# init_filter=64
# latent_dim=64

input_size = 512
patch_size=(4, 4)
init_filter=64
latent_dim=256

aug = None


# if len(sys.argv)>2:
#     config = load_config(sys.argv[2])
#     print_config(config)
#     AUGT = DataAugmentationTransform((input_size, input_size))
#     pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"])
#     spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
#     aug = A.Compose([
#         A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]), 
#         A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
#     ], p=config["augmentation"]["p"])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


best_model_path = f'/cabinet/afshin/base_seg/saved_model/model_{name}'
# --------------------------------------------------------------
data_dir = "/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/dataset/Retinal/DRIVE"
data_dir = "/cabinet/dataset/Retinal/DRIVE"
get_data = partial(get_drive,
                   loading="fast",                
                   data_dir=data_dir,
                   size=(input_size, input_size),
                   tr_te_p=0.7,
                   tr_vl_p=0.75,
                   one_hot=False,
                   tr_dataloader_args={"batch_size": 2, "shuffle": True},
                   vl_dataloader_args={"batch_size": 2, "shuffle": True},
                   te_dataloader_args={"batch_size": 2, "shuffle": True},
                   force_prepare=False,
                   verbose=False,
                   logger=None)

data = get_data(aug=aug)
tr_dataloader = data["tr"]["dataloader"]
vl_dataloader = data["vl"]["dataloader"]

data = get_data(aug=None)
te_dataloader = data["te"]["dataloader"]
# --------------------------------------------------------------


# model = TSegDiff(
#     input_hw=(input_size, input_size),
#     in_ch=1,
#     out_ch=1,
#     init_filter=init_filter,
#     patch_size=patch_size,
#     latent_dim=latent_dim,
# )


model = UNet(
    out_channels=1, 
    in_channels=1, 
    depth=5,
    start_filts=64
)


# model = DAEFormer(
#     num_classes=1,
#     head_count=1, 
#     token_mlp_mode="mix_skip"
# )


total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {total_params}")


if sys.argv[1] == "tr": # --------------- TRAIN ----------------
    print("Trainig")

#     loss_fn = nn.CrossEntropyLoss()
    bce_fn = nn.BCEWithLogitsLoss().to(device)
#     loss_fn = bce_fn
    dice_fn = BinaryDiceLoss().to(device)
    def loss_fn(pd, gt):
        return 0.0*dice_fn(pd, gt) + 0.99*bce_fn(pd, gt) 
    
    if len(sys.argv)<3:
        try:
            checkpoint = torch.load(best_model_path, map_location="cpu")
            model.load_state_dict(checkpoint)
            print("LOADED PRE-TRAINED Ws")
        except:
            print("OOOOOPPPPSSSS")
        

    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
#     optimizer = optim.SGD(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=9)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, last_epoch=epochs, verbose=True)
    # step_lr_schedule = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # -------------------------------------------------------

    for epoch in range(EPOCHS):
        if kill==True: exit()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, model, optimizer, loss_fn, tr_dataloader, writer, device)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(vl_dataloader):
                vimgs, vmsks = vdata["image"].to(device), vdata["mask"].to(device)

                voutputs = model(vimgs)
                
                vloss = loss_fn(voutputs, vmsks)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'ep {epoch_number+1:03d}/{EPOCHS:03d}:-> loss-tr: {avg_loss:.6f}, loss-vl: {avg_vloss:.6f}')

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            print(f"found a better model. prev-vl:{best_vloss:.6f}, curr-vl:{avg_vloss:.6f}")
            best_vloss = avg_vloss
            
            
            try:
                torch.save(model.state_dict(), best_model_path)
            finally:
                if kill==True:
                    exit()
            

        epoch_number += 1


        
else: #----------- TEST -------------
    print("Testing")
    
    try:
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        print("LOADED PRE-TRAINED Ws")
    except:
        print("OOOOOPPPPSSSS")
    model.to(device)
    
    test_metrics = get_binary_metrics()
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(te_dataloader)):
            if kill==True: exit()
            imgs, msks = data["image"].to(device), data["mask"].to(device)
            print(imgs.shape)
        

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

