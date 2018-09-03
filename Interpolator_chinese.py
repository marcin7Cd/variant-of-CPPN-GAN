import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import torch
import torch.autograd as autograd
import gan_cppn_chinese as chinese
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
    
BATCH_SIZE = 150 # Batch size
LATENT_DIM = 128
SINGLE = True
'''
Create image samples and gif interpolations of trained models
'''

if SINGLE:
    noise = torch.randn(BATCH_SIZE, LATENT_DIM)
else:
    noise = torch.randn(BATCH_SIZE,DISP_SIZE, LATENT_DIM)
    
def interpolate(state_dict, generator, preview = True, interpolate = False, large_sample=False,
                disp_size=6, large_dim=256,
                samples=[random.randint(0, BATCH_SIZE - 1), random.randint(0, BATCH_SIZE - 1)],
                counter = 1):
    global noise
    """
    Args:
        
    state_dict: saved copy of trained params
    generator: generator model
    preview: show preview of images in grid form in original size
    interpolate: create interpolation gif
    large_sample: create single images 
    disp_size: size of a grid to show images
    large_dim: dimensions of an individual picture (powers of 2)
    samples: indices of the samples you want to interpolate
    """
    
    c_d = 1
    position = 4
    x_large = large_dim
    y_large = large_dim
    x, y, r = chinese.get_coordinates(x_large//2 + 2*x_large//64, y_large//2+ 2*y_large//64, batch_size=1, scale = 8)
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
        r = r.cuda()
    
    
    generator_int = generator
    generator_int.load_state_dict(torch.load(state_dict, map_location=lambda storage, loc: storage))
    if x_large>64:
        new_conv = torch.nn.Conv2d(64, 1, (5*x_large//64, 5*y_large//64))
        print(generator.conv_seq[1].weight.shape)
        new_conv.weight = torch.nn.Parameter(torch.nn.functional.upsample(generator.conv_seq[1].weight,
                                                    size=(5*x_large//64, 5*y_large//64),
                                                    mode = 'nearest')/((x_large//64)**2))
        generator.conv_seq[1] = new_conv
        print(generator.conv_seq[1].weight.shape)
    if use_cuda:
        generator_int.cuda(gpu)
    
    print(noise.shape)
    if preview:
        noisev = noise[:disp_size**2]
        ones = torch.ones(disp_size**2, x.shape[1], c_d)
        seed = torch.bmm(ones, noisev.unsqueeze(1))
        if use_cuda:
            ones = ones.cuda()
            seed = seed.cuda()
        gen_imgs = generator_int(x, y, r, seed)
        
        gen_imgs = gen_imgs.cpu().data.numpy()
        lib.save_images.save_images(
            gen_imgs,
            'generated_img/samples_chines{}_disp{}_{}x{}.png'.format(counter,
                                                                     disp_size,
                                                                     x_large,
                                                                     x_large)
            
        )

    elif large_sample:
        ones = torch.ones(1, x.shape[1], 1)
        seed = torch.bmm(ones, noise[samples[0],:].unsqueeze(0).unsqueeze(1))
        if use_cuda:
            ones = ones.cuda()
            seed = seed.cuda()
        
        gen_imgs = generator_int(x, y, r, seed)
        gen_imgs = gen_imgs.cpu().data.numpy()
        
        lib.save_images.save_images(
            gen_imgs,
            'generated_img/large_sample_chines{}_samp{}_{}x{}.png'.format(counter,
                                                                          samples[0],
                                                                          x_large,
                                                                          x_large)
        )
    elif interpolate:
               
        nbSteps = 12
        initial_noise = torch.randn((disp_size**2)*2,1,LATENT_DIM)

        to_take=[0,1,6,8,10,11,12,13,14]
        for i in range(15,15+disp_size**2-9):
             to_take= to_take +[i]
        to_take=to_take[:(disp_size**2)]
        initial_noise = torch.cat([initial_noise[x] for x in to_take])
        alphaValues = np.linspace(0, 1, nbSteps)[1:]
        images = []
        if use_cuda:
            noise = noise.cuda(gpu)
            initial_noise = initial_noise.cuda()
        
        samples.append(samples[0])
            
        
        for i in range(len(samples) - 1):
            for alpha in alphaValues:                    
                vector = noise[samples[i]]*(1-alpha) + noise[samples[i + 1]]*alpha
                vector = (vector.unsqueeze(0) + initial_noise)/2
                print(x.shape,vector.unsqueeze(1).shape)
                ones = torch.ones(disp_size**2, x.shape[1], 1)
                
                if use_cuda:
                    ones = ones.cuda()
                print(ones.shape, vector.unsqueeze(1).shape )
                seed = torch.bmm(ones, vector.unsqueeze(1))
                print(seed.shape)
                with torch.no_grad():
                    gen_imgs = generator_int(x, y, r, seed)
                print(gen_imgs.shape)
                if c_d == 3:
                    gen_img_np = np.transpose(gen_imgs.data[0].numpy())
                elif c_d == 1:
                    gen_img_np = gen_imgs.data.cpu().numpy()
                    gen_img_np = gen_img_np.reshape((disp_size**2, x_large-x_large//64+1, y_large-y_large//64+1,1),order = 'C')
                    print(gen_img_np.shape)
                    gen_img_np = gen_img_np.reshape((disp_size, disp_size,gen_img_np.shape[1],gen_img_np.shape[2]),order='C')
                    #gen_img_np = gen_img_np[0]#[0]
                    print(gen_img_np.shape)
                    gen_img_np = np.transpose(gen_img_np,[0,2,1,3]).reshape((gen_img_np.shape[0],gen_img_np.shape[2],
                                                                            gen_img_np.shape[1]*gen_img_np.shape[3]),order='C') \
                                                                   .reshape((gen_img_np.shape[0]*gen_img_np.shape[2],
                                                                            gen_img_np.shape[1]*gen_img_np.shape[3]),order='C')
                    print(gen_img_np.shape)
                images.append(gen_img_np)
            
        
        imageio.mimsave('generated_img/movie_chines{}im{}_{}x{}.gif'.format(counter,
                                                                            disp_size,
                                                                            large_dim,
                                                                            large_dim),
                        images)
        print('saved')
        
        
        
if __name__ == "__main__":
    
    G = chinese.Generator(x_dim = 64, y_dim = 64, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
    for counter in range(21,22):
        interpolate("tmp/chinese/G-cppn-wgan_"+str(counter*10000)+".pth", G,
                    preview=False, large_sample=True,
                    disp_size=6,
                    interpolate=True, large_dim=256,
                    samples=[60,5,7,8,11,19,21,28,31],counter=counter)  
