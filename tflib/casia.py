import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import PIL
import os

directory=os.path.dirname( os.path.dirname(os.path.abspath(__file__)))
directory = os.path.join(directory,"dataset")
print(directory)
def load_casia(bsize, path, resize):
    dataset = datasets.ImageFolder(directory,
                                   transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor()
                                   ]))
    print(dataset)
    loader  = DataLoader(dataset, batch_size=bsize, shuffle=True,pin_memory = True)#,num_workers = 1)
    return loader

        
