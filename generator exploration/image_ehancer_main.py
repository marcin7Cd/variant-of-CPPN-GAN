#coding=utf-8
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2
import torch
import torch.nn as nn
import gan_cppn_chinese as model
import matplotlib.pyplot as plt
import time
import imageio
DISP_X = 8
DISP_Y = 8

ITERATIONS = 300
LEARNING_RATE = 1e-2
SPACE_DIM = 8 #dimensionality of random subspace in which latent space is optimized

use_cuda = torch.cuda.is_available()

class Image_optim(nn.Module):
    def __init__(self, starting_image):
        super(Image_optim, self).__init__()
        self.image = nn.Parameter(starting_image)

    def forward(self, discriminator):
        return discriminator(self.image)

IMAGE_SIZE = 64
x_d = IMAGE_SIZE
y_d = IMAGE_SIZE
OFFSET = -IMAGE_SIZE//2+2
LATENT_DIM = 128
x, y, r = model.get_coordinates(x_d+OFFSET, y_d+OFFSET, batch_size=DISP_X*DISP_Y)
if use_cuda:
    x = x.cuda()
    y = y.cuda()
    r = r.cuda()

ones = -torch.ones(DISP_X*DISP_Y)
if use_cuda:
    ones = ones.cuda()
#creates a projection matrix for subspace 
space = np.random.randn(DISP_X*DISP_Y, LATENT_DIM, SPACE_DIM)

projection_matrix = space @ np.linalg.inv(space.transpose(0,2,1) @ space) \
                    @ space.transpose(0,2,1)

projection_matrix = torch.Tensor(projection_matrix).cuda()

#creates latent position
images = Image_optim(torch.randn(DISP_X*DISP_Y, 1, LATENT_DIM).cuda())

#creates generator and discriminator 
netG = model.Generator(x_dim = x_d, y_dim = y_d, z_dim=LATENT_DIM, batch_size = DISP_X*DISP_Y)
netD = model.Discriminator()
netG.load_state_dict(torch.load('G-cppn-wgan_209000.pth', map_location=lambda storage, loc:storage))
netD.load_state_dict(torch.load('D-cppn-wgan_209000.pth', map_location=lambda storage, loc:storage))
if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    images = images.cuda()

for p in netD.parameters():
    p.requires_grad = False
for p in netG.parameters():
    p.requires_grad = False
for p in images.parameters():
    p.requires_grad = True
    
function = lambda seed : netD(netG(x,y,r,seed))

for i in range(ITERATIONS): #optimization loop
    netD.zero_grad()
    netG.zero_grad()
    images.zero_grad()

    result = images(function)
    result.backward(ones)
    gradient = images.image.grad
    with torch.no_grad():
        images.image -= LEARNING_RATE*gradient @ projection_matrix
    if (i % (ITERATIONS//100) == 0):
        print("{} {}%".format(i ,i*100//ITERATIONS))

#displays result
with torch.no_grad():
    result = netG(x,y,r,images.image)*255
result = result.cpu().numpy()
result = np.reshape(result, (DISP_X,DISP_Y,IMAGE_SIZE,IMAGE_SIZE))
result = np.transpose(result, [0,2,1,3])
result = np.reshape(result,
                    (result.shape[0]*result.shape[1],
                     result.shape[2]*result.shape[3],1))
cv2.imwrite('enhanced_images.png',result)
