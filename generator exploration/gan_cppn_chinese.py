import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import os, sys
sys.path.append(os.getcwd())

import time
import math
import matplotlib
#matplotlib.use('Agg')

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
CUDA_LAUNCH_BLOCKING=1
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

IMAGE_SIZE = 64
DIM = 64 # Model dimensionality
BATCH_SIZE = 64 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
GEN_ITER = 1
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 260000 # How many generator iterations to train for
OUTPUT_DIM = IMAGE_SIZE*IMAGE_SIZE 
LATENT_DIM = 128

OFFSET = -IMAGE_SIZE//2+2

# ==================CPPN Modifications======================

def get_coordinates(x_dim = 28, y_dim = 28, scale = 8, batch_size = 1):
    '''
    calculates and returns a vector of x and y coordinates, and corresponding radius from the centre of image.
    '''
    n_points = x_dim * y_dim

    # creates a list of x_dim values ranging from -1 to 1, then scales them by scale
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5        
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), 1).reshape(1, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), 1).reshape(1, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), 1).reshape(1, n_points, 1)
    
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()

class Generator(nn.Module):
    def __init__(self, x_dim, y_dim, batch_size=1, z_dim = 32, c_dim = 1, scale = 8.0, net_size = 128, devid = -1,):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.net_size = net_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.scale = scale
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.scale = scale
        
        #Build NN graph
        self.linear1 = nn.Linear(z_dim, self.net_size*2)
        self.linear2 = nn.Linear(1, self.net_size*2, bias=False)
        self.linear3 = nn.Linear(1, self.net_size*2, bias=False)
        self.linear4 = nn.Linear(1, self.net_size*2, bias=False)
        
        self.linear5 = nn.Linear(self.net_size*2, self.net_size*2)
        self.linear6 = nn.Linear(self.net_size*2, self.net_size)
        self.linear7 = nn.Linear(self.net_size, self.net_size)
        self.linear8 = nn.Linear(self.net_size, self.net_size//2)
        
        self.linear9 = nn.Linear(self.net_size//2, self.net_size//2)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1,False)
        #self.softplus = nn.Softplus()
        
        self.lin_seq = nn.Sequential(self.leaky_relu, self.linear5,
                                     self.leaky_relu, self.linear5,
                                     self.leaky_relu, self.linear6,
                                     self.leaky_relu, self.linear7,
                                     self.leaky_relu, self.linear7,
                                     self.leaky_relu, self.linear8,
                                     self.leaky_relu, self.linear9,
                                     self.leaky_relu, self.linear9,
                                     self.leaky_relu)
        self.conv_seq = nn.Sequential(
            nn.Upsample(scale_factor=2),
            #nn.ReLU(True),
            nn.Conv2d(self.net_size//2, 1, 5),
            self.sigmoid)
        
    def forward(self, x, y, r, z):

        # self.scale * z?
        U = self.linear4(r) + self.linear2(x) + self.linear3(y)
        U = U + self.linear1(z) 
        #print(U.shape, x.size())
        
        result = self.lin_seq(U).transpose(1,2).view(z.size()[0],-1, math.sqrt(x.size()[1]), math.sqrt(x.size()[1]))
        #print(result.shape)
        result = self.conv_seq(result)
        #print(result.shape)
        '''
        result = self.lin_seq(U)
        size_x = int(math.sqrt(x.size()[0]))
        result = self.lin_seq(U)
        return result.squeeze(1).view(size_x,size_x )#.view(-1, result.size()[1]*result.size()[2])
        '''
        return result.view(-1, result.size()[2]*result.size()[3])
        


def get_mask(dim_x,dim_y,kernel, padding=(0,0), stride=(1,1)):
    kernel_x, kernel_y = kernel
    padding_x, padding_y = padding
    stride_x, stride_y = stride
    test_x = torch.ones(1,1,(dim_x+2*padding_x-kernel_x)//stride_x+1)
    result_x = torch.nn.functional.conv_transpose1d(test_x,torch.ones(1,1,kernel_x),
                                  stride = stride_x,
                                  padding = padding_x)
    offset_x = (dim_x+2*padding_x+1)% stride_x
    if offset_x != 0:
        result_x = torch.cat([result_x, torch.ones(1,1,offset_x)],
                             dim=2)

    test_y = torch.ones(1,1,(dim_y+2*padding_y-kernel_y)//stride_y+1)
    result_y = torch.nn.functional.conv_transpose1d(test_y,torch.ones(1,1,kernel_y),
                                  stride = stride_y,
                                  padding = padding_y)
    offset_y = (dim_y+2*padding_y+1)% stride_y
    if offset_y != 0:
        result_y = torch.cat([result_y, torch.ones(1,1,offset_y)],
                             dim=2)
    return ((kernel_x-1)//stride_x+1)*((kernel_y-1)//stride_y+1)/torch.mul(result_x.unsqueeze(3),result_y.unsqueeze(2))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(False),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(False),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(False),
            nn.Conv2d(4*DIM, 8*DIM, 5, stride=2, padding=2),
            nn.ReLU(False),
        )
        self.main = main
        self.output = nn.Linear(8*4*4*DIM, 1)
        ##masks for 1st convolution
        self.mask_conv1_pre = get_mask(IMAGE_SIZE,
                                       IMAGE_SIZE,
                                       (5,5),(2,2),(2,2))
        self.mask_conv1_post = 1/torch.nn.functional.conv2d(self.mask_conv1_pre,torch.ones(1,1,5,5),stride=2, padding=2)

        ##masks for 2nd convolution
        self.mask_conv2_pre = get_mask(self.mask_conv1_post.shape[2],
                                       self.mask_conv1_post.shape[3],
                                       (5,5),(2,2),(2,2))
        self.mask_conv2_post = 1/torch.nn.functional.conv2d(self.mask_conv2_pre,torch.ones(1,1,5,5),stride=2, padding=2)

        ##masks for 3rd convolution
        self.mask_conv3_pre = get_mask(self.mask_conv2_post.shape[2],
                                       self.mask_conv2_post.shape[3],
                                       (5,5),(2,2),(2,2))
        self.mask_conv3_post = 1/torch.nn.functional.conv2d(self.mask_conv3_pre,torch.ones(1,1,5,5),stride=2, padding=2)

        ##masks for 4th convolution
        self.mask_conv4_pre = get_mask(self.mask_conv3_post.shape[2],
                                       self.mask_conv3_post.shape[3],
                                       (5,5),(2,2),(2,2))
        self.mask_conv4_post = 1/torch.nn.functional.conv2d(self.mask_conv4_pre,torch.ones(1,1,5,5),stride=2, padding=2)
        
        print(1,self.mask_conv1_pre.shape)
        print(2,self.mask_conv1_post.shape)
        print(3,self.mask_conv2_pre.shape)
        print(4,self.mask_conv2_post.shape)
        print(5,self.mask_conv3_pre.shape)
        print(6,self.mask_conv3_post.shape)
        print(7,self.mask_conv4_pre.shape)
        print(8,self.mask_conv4_post.shape)
        
        self.main[0].weight.data = self.main[0].weight/self.mask_conv1_pre.mean()
        self.main[2].weight.data = self.main[2].weight/self.mask_conv2_pre.mean()
        self.main[4].weight.data = self.main[4].weight/self.mask_conv3_pre.mean()
        self.main[6].weight.data = self.main[6].weight/self.mask_conv4_pre.mean()
        if use_cuda:
            self.mask_conv1_pre = self.mask_conv1_pre.cuda()
            self.mask_conv1_post = self.mask_conv1_post.cuda()
            self.mask_conv2_pre = self.mask_conv2_pre.cuda()
            self.mask_conv2_post = self.mask_conv2_post.cuda()
            self.mask_conv3_pre = self.mask_conv3_pre.cuda()
            self.mask_conv3_post = self.mask_conv3_post.cuda()
            self.mask_conv4_pre = self.mask_conv4_pre.cuda()
            self.mask_conv4_post = self.mask_conv4_post.cuda()
        

    def forward(self, input):
        result = input.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        #out = self.main(input)
        result = self.main[0](result*self.mask_conv1_pre)
        result = self.main[1](result*self.mask_conv1_post)
        
        result = self.main[2](result*self.mask_conv2_pre)
        result = self.main[3](result*self.mask_conv2_post)
        
        result = self.main[4](result*self.mask_conv3_pre)
        result = self.main[5](result*self.mask_conv3_post)
        
        result = self.main[6](result*self.mask_conv4_pre)
        result = self.main[7](result*self.mask_conv4_post)
        
        out = result.view(-1, 8*4*4*DIM)
        out = self.output(out)
        return out.view(-1)
