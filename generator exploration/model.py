import os, sys
sys.path.append(os.getcwd())
import torch
import torch.autograd as autograd
import gan_cppn_casia_big as casia
import numpy as np
import imageio
import random

torch.manual_seed(111)
use_cuda = torch.cuda.is_available()
print(use_cuda)
LATENT_DIM = 128
IMAGE_SIZE = 64
OFFSET = -IMAGE_SIZE//2+2
class Model():
    def __init__(self, dim, generator_file):
        self.noise_dim = LATENT_DIM
        self.generator = casia.Generator(x_dim = IMAGE_SIZE,
                                         y_dim = IMAGE_SIZE,
                                         z_dim=LATENT_DIM,
                                         batch_size = dim)
        self.generator.load_state_dict(torch.load(generator_file,
                                                  map_location=lambda storage,
                                                  loc: storage))
        x, y, r = casia.get_coordinates(IMAGE_SIZE+OFFSET,
                                        IMAGE_SIZE+OFFSET,
                                        batch_size=dim,scale = 8)
        self.x = x
        self.y = y
        self.r = r
        if use_cuda:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
            self.r = self.r.cuda()
            self.generator = self.generator.cuda(0)
        
    def generate_image(self,variable):
        with torch.no_grad():
            result = self.generator(self.x,
                                    self.y,
                                    self.r,
                                    torch.FloatTensor(variable).unsqueeze(1).cuda())
        return result.reshape(-1,1,IMAGE_SIZE,IMAGE_SIZE).cpu().numpy()
    
