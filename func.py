import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch

def plotar(imgs):
    plt.rcParams["savefig.bbox"] = 'tight'
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()




from torchvision.io import read_image
def ler_imagem(dir:str):
    return read_image(dir)


from torchvision.utils import make_grid
def juntar_fotos(lista:list):
    return make_grid(lista)


from torchvision.utils import draw_bounding_boxes
criar_caixa = draw_bounding_boxes
