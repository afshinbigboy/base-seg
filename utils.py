import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


jet = plt.get_cmap("jet")


def write_imgs(
    writer,
    imgs, msks, prds, 
    step, id, dataset, 
    ids=None
):
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
