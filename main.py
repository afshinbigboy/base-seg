import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from functools import partial
import os, sys, json
from dsl.datasets import get_drive
from losses import BinaryDiceLoss
from models.trans import MTv00
from models.unet import UNet
from models.tseg import *
# from models.daeformer import DAEFormer
# from other_models import MERIT_Cascaded
from trainer import train_one_epoch
from metrics import get_binary_metrics
from utils import write_imgs, load_config, print_config
from augs import DataAugmentationTransform, A
from common.context_manangers import safezone
from arguments import get_parser



args = get_parser().parse_args()

EPOCHS = 1000
LR=0.01
batch_size = 2
input_size = 1024
patch_size = (8, 8)
init_filter = 64
latent_dim = 256


aug = None
best_vloss = np.Inf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = f'{args.output_dir}/model_{args.run_name}'

# os.system(f"rm -rf runs/bseg_{args.run_name}")
writer = SummaryWriter(f'runs/bseg_{args.run_name}')


# >>>>>>>>>>>>>>>>> Preparing dataloaders >>>>>>>>>>>>>>>>>>
get_data = partial(get_drive,
                   loading="fast",                
                   data_dir=args.data_dir,
                   size=(input_size, input_size),
                   tr_te_p=0.7,
                   tr_vl_p=0.75,
                   one_hot=False,
                   tr_dataloader_args={"batch_size": batch_size, "shuffle": True},
                   vl_dataloader_args={"batch_size": batch_size},
                   te_dataloader_args={"batch_size": batch_size},
                   force_prepare=False,
                   verbose=False,
                   logger=None)

data = get_data(aug=aug)
tr_dataloader = data["tr"]["dataloader"]
vl_dataloader = data["vl"]["dataloader"]

data = get_data(aug=None)
te_dataloader = data["te"]["dataloader"]
# <<<<<<<<<<<<<<<<< Preparing dataloaders <<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>> Preparing the Model >>>>>>>>>>>>>>>>>>
model = TSegDiff(
    input_hw=(input_size, input_size),
    in_ch=1,
    out_ch=1,
    init_filter=init_filter,
    patch_size=patch_size,
    latent_dim=latent_dim,
)

# model = UNet(
#     out_channels=1, 
#     in_channels=1, 
#     depth=5,
#     start_filts=64
# )

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {total_params}")
# <<<<<<<<<<<<<<<<< Ending the Model <<<<<<<<<<<<<<<<<<



# --------------- TRAIN ----------------
if args.train: 
    print("Trainig")

    # --> prepare loss function
    bce_fn = nn.BCEWithLogitsLoss().to(device)
    dice_fn = BinaryDiceLoss().to(device)
    def loss_fn(pd, gt):
        return 0.0*dice_fn(pd, gt) + 0.99*bce_fn(pd, gt)
    
    # --> load pre-weights
    if args.countinue:
        try:
            checkpoint = torch.load(best_model_path, map_location="cpu")
            model.load_state_dict(checkpoint)
            model = model.to(device)
            print("Loaded pre-trained weights.")
        except Exception as e:
            print(f"There is not compatible weights to continue training...\n{e}")

    # --> prepare optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=9)
    
    # --> start training
    for epoch in range(EPOCHS):
        model.train(True)
        avg_loss = train_one_epoch(epoch, model, optimizer, loss_fn, tr_dataloader, writer, device)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(vl_dataloader):
                vimgs, vmsks = vdata["image"].to(device), vdata["mask"].to(device)
                voutputs = model(vimgs)
                vloss = loss_fn(voutputs, vmsks)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'ep {epoch+1:03d}/{EPOCHS:03d}:-> loss-tr: {avg_loss:.6f}, loss-vl: {avg_vloss:.6f}')

        writer.add_scalars('Training vs. Validation Loss', {'Training':avg_loss, 'Validation':avg_vloss}, epoch+1)
        writer.flush()

        if avg_vloss < best_vloss:
            print(f"found a better model. prev-vl:{best_vloss:.6f}, curr-vl:{avg_vloss:.6f}")
            best_vloss = avg_vloss
            
            with safezone():
                torch.save(model.state_dict(), best_model_path)

#----------- TEST -------------
elif args.test: 
    print("Testing")
    try:
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        print("Loaded trained weights.")
    except:
        print("There is a problem on loading weigths!")
        sys.exit(1)
        
    model.to(device)
    test_metrics = get_binary_metrics()
    model.eval()
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
            write_imgs(writer, imgs, msks, prds, i, args.run_name, dataset="drive".upper())

    result = test_metrics.compute()
    writer.add_scalars(
        f"Metrics/test-{args.run_name}",
        result,
    )    
    print(json.dumps({k: v.item() for k, v in result.items()}, indent=4))

else:
    print("no train ... no test !")
