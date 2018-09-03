import os, sys
sys.path.append(os.getcwd())
import torch
import torch.autograd as autograd
import gan_cppn_mnist3 as mnist
from functools import reduce
import numpy as np
import imageio
import random

torch.manual_seed(111)
use_cuda = torch.cuda.is_available()
print(torch.cuda.device_count(),"-----------")
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_name(0))
print('')
print('')
if use_cuda:
    gpu = 0
    
BATCH_SIZE = 50 # Batch size
DISP_SIZE = 5*4
LATENT_DIM = 128#128
ONE_HOT_SIZE=10
'''
Create image samples and gif interpolations of trained models
'''
noise = torch.randn(BATCH_SIZE, LATENT_DIM-6*ONE_HOT_SIZE)

def interpolate(state_dict, generator,
                large_dim=64, combinations=[[(0,0)]],
                samples=[random.randint(0, BATCH_SIZE - 1), random.randint(0, BATCH_SIZE - 1)],
                counter = 1):
    global noise
    """
    Args:
        
    state_dict: saved copy of trained params
    generator: generator model
    samples: indices of the samples you want to interpolate
    combinations: list of proportions in which categories have to mixed
    """
    one_hot = torch.diag(torch.ones([ONE_HOT_SIZE]))
    one_hot = torch.cat([one_hot,one_hot,one_hot,one_hot,one_hot,one_hot],dim=1)
    if use_cuda:
        one_hot = one_hot.cuda()
    for mix in combinations:
        image_seed = torch.zeros_like(one_hot[0])
        for contrib in mix:
            image_seed += one_hot[contrib[0]]*contrib[1]
        one_hot = torch.cat([one_hot,image_seed.unsqueeze(0)],dim=0)
    c_d = 1
    position = 4
    x, y, r = mnist.get_coordinates(large_dim, large_dim, batch_size=BATCH_SIZE,scale = 8)
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
        r = r.cuda()
        #y = y*0.8-r*0.2
        #x = x*0.8-r*0.2
        #r = r*0.8
    x_large = large_dim
    y_large = large_dim
    
    
    generator_int = generator
    generator_int.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage))
    print(noise.shape)
    if use_cuda:
        generator_int = generator_int.cuda()
    nbSteps = 10
    alphaValues = np.linspace(0, 1, nbSteps)[1:]
    images = []
    if use_cuda:
        noise = noise.cuda(gpu)
    x, y, r = mnist.get_coordinates(x_large, y_large, batch_size=1, scale = 8)
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
        r = r.cuda()
        
    samples.append(samples[0])
    onesb=torch.ones(DISP_SIZE,1).cuda()
        
    for i in range(len(samples) - 1):  
        for alpha in alphaValues:                    
            vector = noise[samples[i]].unsqueeze(0)*(1-alpha) + noise[samples[i + 1]].unsqueeze(0)*alpha
            vector = torch.matmul(onesb,vector)
            print(i)
                
            ones = torch.ones(DISP_SIZE, x.shape[1], 1)
            if use_cuda:
                ones = ones.cuda()
            seed = torch.cat([torch.bmm(ones, vector.unsqueeze(1)),
                              torch.bmm(ones, one_hot.unsqueeze(1))],dim=2)
            if use_cuda:
                seed = seed.cuda()
            with torch.no_grad():
                gen_imgs = generator_int(x, y, r, seed)
            if c_d == 3:
                gen_img_np = np.transpose(gen_imgs.data[0].numpy())
            elif c_d == 1:
                gen_img_np = gen_imgs.data.cpu().numpy()
                gen_img_np = gen_img_np.reshape((DISP_SIZE, x_large, y_large,1),order = 'C')
                gen_img_np = gen_img_np.reshape((DISP_SIZE//5,5,gen_img_np.shape[1],gen_img_np.shape[2]),order='C')
                gen_img_np = np.transpose(gen_img_np,[0,2,1,3]).reshape((gen_img_np.shape[0],gen_img_np.shape[2],
                                                                        gen_img_np.shape[1]*gen_img_np.shape[3]),order='C') \
                                                               .reshape((gen_img_np.shape[0]*gen_img_np.shape[2],
                                                                        gen_img_np.shape[1]*gen_img_np.shape[3]),order='C')
                print(gen_img_np.shape)
            images.append(gen_img_np)
            
        
    imageio.mimsave('generated_img/movie_mnist_28.gif', images)
    print('saved')
        
        
        
if __name__ == "__main__":

    category= \
    {"three/two" : 0,
     "five": 1,
     "one": 2,
     "zero": 3,
     "nine": 4,
     "four/nine": 5,
     "six": 6,
     "four": 7,
     "seven": 8,
     "eight": 9}
    G = mnist.Generator(x_dim = 28, y_dim = 28, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
    interpolate("tmp\\mnist3\\G-cppn-wgan_4000.pth", G, large_dim=64, samples=[4,5,7,8,11,19,20,21,23],
                combinations=[[(category["four/nine"],0.5),(category["three/two"],0.5)],
                             [(category["five"],0.5),(category["six"],0.5)],
                             [(category["one"],0.5),(category["four"],0.5)],
                             [(category["zero"],0.5),(category["seven"],0.5)],
                             [(category["nine"],0.5),(category["eight"],0.5)],
                             [(category["four"],0.5),(category["seven"],0.5)],
                             [(category["three/two"],0.5),(category["seven"],0.5)],
                             [(category["nine"],0.5),(category["six"],0.5)],
                             [(category["one"],0.5),(category["four/nine"],0.5)],
                             [(category["four"],0.5),(category["six"],0.5)]
                            ])
