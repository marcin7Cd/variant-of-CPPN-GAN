import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import torch
import torch.autograd as autograd
import gan_cppn_mnist as mnist
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
    
BATCH_SIZE = 100 # Batch size
DISP_SIZE = 9
LATENT_DIM = 128#128
'''
Create image samples and gif interpolations of trained models
'''
noise = torch.randn(BATCH_SIZE, LATENT_DIM)
   
def interpolate(state_dict, generator, preview = True, interpolate = False, large_sample=False,
                disp_size=3, large_dim=1024,
                samples=[random.randint(0, BATCH_SIZE - 1), random.randint(0, BATCH_SIZE - 1)],
                counter = 1):
    global noise
    """
    Args:
        
    state_dict: saved copy of trained params
    generator: generator model
    disp_size: size of a grid
    preview: show preview of images in grid form in original size )
    interpolate: create interpolation gif
    large_sample: create a large sample of an individual picture
    large_dim: dimension of image
    samples: indices of the samples you want to interpolate
    """
    
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
    if preview:
        
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = noise[:disp_size**2]
        print(noisev.shape)
        ones = torch.ones(disp_size**2, x.shape[1], c_d)
        if use_cuda:
            ones = ones.cuda()
               
        seed = torch.bmm(ones, noisev.unsqueeze(1))
        if use_cuda:
            seed = seed.cuda()
        print(seed.shape)
        gen_imgs = generator_int(x, y, r, seed)
        
        gen_imgs = gen_imgs.cpu().data.numpy()
        print(gen_imgs.shape)
        lib.save_images.save_images(
            gen_imgs,
            'generated_img/samples_temp'+str(counter)+'.png'
            
        )

    elif large_sample:
        with torch.no_grad():
            noisev = noise[position]
        ones = torch.ones(1, x.shape[1], 1)
        seed = torch.bmm(ones, noisev.unsqueeze(0).unsqueeze(0))
        if use_cuda:
            ones = ones.cuda()
            seed = seed.cuda()
        gen_imgs = generator_int(x, y, r, seed)
        gen_imgs = gen_imgs.cpu().data.numpy()
        
        lib.save_images.save_images(
            gen_imgs,
            'generated_img/large_sample'+str(counter)+'.png'
        )
    elif interpolate:
               
        nbSteps = 13
        initial_noise = torch.randn((disp_size**2)*2,1,LATENT_DIM)
        to_take=[0,1,6,8,10,11,12,13,14]
        for i in range(15,15+disp_size**2-9):
             to_take= to_take +[i]
        to_take=to_take[:(disp_size**2)]
       
        initial_noise = torch.cat([initial_noise[x] for x in to_take])
        print(initial_noise.shape)
        alphaValues = np.linspace(0, 1, nbSteps)[1:]
        images = []
        if use_cuda:
            noise = noise.cuda(gpu)
            initial_noise = initial_noise.cuda()
        with torch.no_grad():
            noisev = noise[position]
               
        samples.append(samples[0])
            
        
        for i in range(len(samples) - 1):
            for alpha in alphaValues:                    
                print(i,alpha)
                vector = noise[samples[i]]*(1-alpha) + noise[samples[i + 1]]*alpha
                vector = (vector.unsqueeze(0) + initial_noise)/2
                ones = torch.ones(disp_size**2, x.shape[1], 1)
                
                if use_cuda:
                    ones = ones.cuda()
                seed = torch.bmm(ones, vector.unsqueeze(1))
                
                if use_cuda:
                    seed = seed.cuda()
                with torch.no_grad():
                    gen_imgs = generator_int(x, y, r, seed)
                gen_img_np = gen_imgs.data.cpu().numpy()
                gen_img_np = gen_img_np.reshape((disp_size**2, x_large, y_large,1),order = 'C')
                gen_img_np = gen_img_np.reshape((disp_size, disp_size, gen_img_np.shape[1],gen_img_np.shape[2]),order='C')
                gen_img_np = np.transpose(gen_img_np,[0,2,1,3]).reshape((gen_img_np.shape[0],gen_img_np.shape[2],
                                                                        gen_img_np.shape[1]*gen_img_np.shape[3]),order='C') \
                                                                .reshape((gen_img_np.shape[0]*gen_img_np.shape[2],
                                                                        gen_img_np.shape[1]*gen_img_np.shape[3]),order='C')
                images.append(gen_img_np)
            
        
        imageio.mimsave('generated_img/movie_mnist1_{}.gif'.format(x_large), images)
        print('saved')
        
        
        
if __name__ == "__main__":
    
    G = mnist.Generator(x_dim = 28, y_dim = 28, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
    interpolate("tmp\\mnist\\G-cppn-wgan_3000.pth", G,
                preview=False, large_sample=False, 
                interpolate=True, disp_size=10,
                large_dim=64, samples=[13, 9, 5, 1, 20])
