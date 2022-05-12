# Implementação e treinamento da rede
import torch
from torch import nn, optim
from torchvision.transforms.functional import convert_image_dtype

# Carregamento de Dados e Modelos
from torchvision import datasets, models
from torchvision import transforms

# Plots e análises
import matplotlib.pyplot as plt
import numpy as np
import time, os
from func import *

# Configurando hiperparâmetros.
args = {
    'device' : 'cpu'
}
inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# Definindo dispositivo de hardware
if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(args['device'])

outputs = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True,progress=False)


foto_1 = ler_imagem('foto2.jpg')
foto_2 = ler_imagem('foto2.jpg')

batch_int = torch.stack([foto_1, foto_2])
batch = convert_image_dtype(batch_int, dtype=torch.float)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()
torch.cuda.empty_cache()
batch=batch.to(args['device'])
model=model.to(args['device'])
outputs = model(batch)


score_threshold = 0.6
deteccoes=[]
for deteccao, output in zip(batch_int, outputs):
    deteccoes.append(criar_caixa(deteccao, boxes=output['boxes'][output['scores'] > score_threshold], width=2,labels=[inst_classes[c] for c in output['labels'][output['scores'] > score_threshold]],colors=[(0,255,255) for c in output['labels'][output['scores'] > score_threshold]]))
    plotar(deteccoes[-1])

