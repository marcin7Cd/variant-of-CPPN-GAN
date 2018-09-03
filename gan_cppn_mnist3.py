import os, sys
sys.path.append(os.getcwd())

import time
import math
import matplotlib
matplotlib.use('Agg')
import numpy as np

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
    gpu = 0

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
GEN_ITERS = 4
CRITIC_ITERS = 20 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 2 # Gradient penalty lambda hyperparameter
ITERS = 160000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
LATENT_DIM = 128 #dimension of latent variable sample z
D_LEARNING_RATE = 1e-3
G_LEARNING_RATE = 1e-3

IMAGE_SIZE=28

lib.print_model_settings(locals().copy())

# ==================CPPN Modifications======================

def get_coordinates(x_dim = 28, y_dim = 28, scale = 8, batch_size = 1):

    n_points = x_dim * y_dim

    # creates a list of x_dim values ranging from -1 to 1, then scales them by scale
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)*0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)*0.5        
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = x_mat*x_mat - y_mat*y_mat
    x_mat = np.tile(x_mat.flatten(), 1).reshape(1, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), 1).reshape(1, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), 1).reshape(1, n_points, 1)
    
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()
        
x_d = 28
y_d = 28
x, y, r = get_coordinates(x_d, y_d, batch_size=BATCH_SIZE)
if use_cuda:
    x = x.cuda()
    y = y.cuda()
    r = r.cuda()

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
        self.linear1 = nn.Linear(z_dim, self.net_size)
        self.linear_space = nn.Linear(3, self.net_size, bias=False)
        #self.linear3 = nn.Linear(1, self.net_size, bias=False)
        #self.linear4 = nn.Linear(1, self.net_size, bias=False)
        
        self.linear5 = nn.Linear(self.net_size, self.net_size)
        self.linear6 = nn.Linear(self.net_size, self.net_size)
        self.linear7 = nn.Linear(self.net_size, self.net_size)
        self.linear8 = nn.Linear(self.net_size, self.net_size)
        self.linear9 = nn.Linear(self.net_size, self.c_dim)
        
        #self.linear9 = nn.Linear(self.net_size, self.c_dim)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1,False)
        #self.softplus = nn.Softplus()
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=1)
        
        self.lin_seq = nn.Sequential(self.leaky_relu, self.linear5,
                                     self.leaky_relu, self.linear6,
                                     self.leaky_relu, self.linear7,
                                     self.leaky_relu, self.linear8,
                                     self.leaky_relu, self.linear9,
                                     self.hardtanh)

        #self.linear1.weight.data = self.linear1.weight.data \
        #                           * (torch.cat([torch.ones([1,LATENT_DIM-10])*LATENT_DIM/10*(1/4),
        #                                         torch.ones([1,10])*LATENT_DIM/(LATENT_DIM-10)*(3/4)],
        #                                        dim=1))       
    def forward(self, x, y, r, z):

        # self.scale * z?
        space = torch.cat((x,y,r),-1)
        U = self.linear_space(space) + self.linear1(z)   
        result = self.lin_seq(U).squeeze(2).view(z.size()[0], math.sqrt(x.size()[1]), math.sqrt(x.size()[1])).unsqueeze(1)
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
    offset_y = (dim_x+2*padding_x+1)% stride_x
    if offset_y != 0:
        result_y = torch.cat([result_y, torch.ones(1,1,offset_y)],
                             dim=2)
    return ((kernel_x-1)//stride_x+1)*((kernel_y-1)//stride_y+1)/torch.mul(result_x.unsqueeze(3),result_y.unsqueeze(2))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
        
        print(1,self.mask_conv1_pre.shape)
        print(2,self.mask_conv1_post.shape)
        print(3,self.mask_conv2_pre.shape)
        print(4,self.mask_conv2_post.shape)
        print(5,self.mask_conv3_pre.shape)
        print(6,self.mask_conv3_post.shape)
        #print(self.mask4)
        #print(self.mask4)
        
        self.main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(False),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(False),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(False)
            )
        """
        self.mask1 = self.mask1/self.mask1.mean()
        self.mask2 = self.mask2/self.mask2.mean()
        self.mask3 = self.mask3/self.mask3.mean()
        self.mask4 = self.mask4/self.mask4.mean()
        """
        self.main[0].weight.data = self.main[0].weight/self.mask_conv1_pre.mean()
        self.main[2].weight.data = self.main[2].weight/self.mask_conv2_pre.mean()
        self.main[4].weight.data = self.main[4].weight/self.mask_conv3_pre.mean()
        
        if use_cuda:
            self.mask_conv1_pre = self.mask_conv1_pre.cuda()
            self.mask_conv1_post = self.mask_conv1_post.cuda()
            self.mask_conv2_pre = self.mask_conv2_pre.cuda()
            self.mask_conv2_post = self.mask_conv2_post.cuda()
            self.mask_conv3_pre = self.mask_conv3_pre.cuda()
            self.mask_conv3_post = self.mask_conv3_post.cuda()
                    
        self.output = nn.Linear(4*4*4*DIM,1)
        
    def forward(self, input):
        #result = input
        result = input.view(-1,1,IMAGE_SIZE, IMAGE_SIZE)
        result = self.main[0](result*self.mask_conv1_pre)
        result = self.main[1](result*self.mask_conv1_post)
        
        result = self.main[2](result*self.mask_conv2_pre)
        result = self.main[3](result*self.mask_conv2_post)
        
        result = self.main[4](result*self.mask_conv3_pre)
        result = self.main[5](result*self.mask_conv3_post)
        
        result = result.view(-1,4*4*4*DIM)
        result = self.output(result)
        result = torch.sigmoid(result)
        return result.view(-1)


"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        

        self.in_features = 4*DIM*4*4
        self.out_features = 16
        self.kernel_dims = 32
        '''
        self.mean = True
        self.T = nn.Parameter(torch.Tensor(self.in_features,
                                           self.out_features,
                                           self.kernel_dims))
        torch.nn.init.normal_(self.T, 0, 1)
        '''
        self.output = nn.Linear(4*4*4*DIM, 1)
        
    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        x = self.main(input)
        x = x.view(-1, 4*4*4*DIM)
        x = self.output(x)
        # #### Minibatch Discrimination ###
        '''
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        norm = -norm
        max_norm = torch.max(norm,0)[0]
        #print(max_norm.size())
        norm = norm-max_norm
        expnorm = torch.exp(norm)
        o_b = (torch.exp(max_norm)*expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1
        x = o_b
        #print(x.size())
        #x = torch.cat([x, o_b], 1)
        '''
        return x#out.view(-1)
"""
def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, LATENT_DIM-60)
    one_hot = categories.sample([BATCH_SIZE])
    one_hot = torch.cat([one_hot,one_hot,one_hot,one_hot,one_hot,one_hot],dim=1)
    noise = torch.cat([noise,
                       one_hot],dim=1)
            
    if use_cuda:
        noise = noise.cuda(gpu)
################################3
    with torch.no_grad():
        noisev = autograd.Variable(noise)#, volatile=True)
    
        ones = torch.ones(BATCH_SIZE, 28 * 28, 1)
        if use_cuda:
            ones = ones.cuda()
            
        seed = torch.bmm(ones, noisev.unsqueeze(1))
    
        samples = netG(x, y, r, seed)
        samples = samples.view(BATCH_SIZE, 28, 28)
        # print samples.size()

        samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        'tmp/mnist3/samples_{}.png'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

if __name__ == "__main__":
    netG = Generator(x_dim = x_d, y_dim = y_d, z_dim=LATENT_DIM, batch_size = BATCH_SIZE)
    netD = Discriminator()
    #netG.load_state_dict(torch.load("tmp\\mnist3\\G-cppn-wgan_2000.pth", map_location=lambda storage, loc: storage))
    #netD.load_state_dict(torch.load("tmp\\mnist3\\D-cppn-wgan_2000.pth", map_location=lambda storage, loc: storage))
    print (netG)
    print (netD)
    
    if use_cuda:
        netD = netD.cuda(gpu)
        netG = netG.cuda(gpu)
    
    optimizerD = optim.Adam(netD.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.9))
    
    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)
    
    data = inf_train_gen()

    categories = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    ######
    
    for iteration in range(0,ITERS):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        #for p in netG.parameters():  # reset requires_grad
        #    p.requires_grad = False  # they are set to False below in netG update
            
        for iter_d in range(CRITIC_ITERS):
            
            _data = next(data)
            real_data = torch.Tensor(_data)
            if real_data.size()[0] == BATCH_SIZE:
    
                if use_cuda:
                    real_data = real_data.cuda(gpu)
                real_data_v = autograd.Variable(real_data)
                netD.zero_grad()
                netG.zero_grad()
                
                # train with real
                D_real = netD(real_data_v)
                D_real = D_real.mean()
                
                # print D_real
                D_real.backward(mone)
                # train with fake
                noise = torch.randn(BATCH_SIZE, LATENT_DIM-60)
                one_hot = categories.sample([BATCH_SIZE])
                one_hot = torch.cat([one_hot,one_hot,one_hot,one_hot,one_hot,one_hot],dim=1)
                noise = torch.cat([noise,
                                   one_hot],dim=1)
    
                if use_cuda:
                    noise = noise.cuda(gpu)

                with torch.no_grad():
                    noisev= autograd.Variable(noise)
                    
                ones = torch.ones(BATCH_SIZE, 28 * 28, 1)
                if use_cuda:
                    ones = ones.cuda()
                    
                seed = torch.bmm(ones, noisev.unsqueeze(1))
                with torch.no_grad():
                    #fake = autograd.Variable(netG(x, y, r, seed).data)
                    fake = netG(x, y, r, seed).data
                inputv = fake
                D_fake = netD(inputv)
                D_fake = D_fake.mean()
                D_fake.backward(one)
        
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
                gradient_penalty.backward()
        
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake

                netG.zero_grad()
                
                optimizerD.step()
                del D_fake
                del D_real
                del gradient_penalty
    
        ############################
        # (2) Update G network
        ###########################

        for iter_g in range(GEN_ITERS):
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            #for p in netG.parameters():  # reset requires_grad
            #    p.requires_grad = True  # they are set to False below in netG update
            netG.zero_grad()
            netD.zero_grad()
            noise = torch.randn(BATCH_SIZE, LATENT_DIM-60)
            one_hot = categories.sample([BATCH_SIZE])
            one_hot = torch.cat([one_hot,one_hot,one_hot,one_hot,one_hot,one_hot],dim=1)
            noise = torch.cat([noise,
                               one_hot],dim=1)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise)
        
            ones = torch.ones(BATCH_SIZE, 28 * 28, 1)
            if use_cuda:
                ones = ones.cuda()
                
            seed = torch.bmm(ones, noisev.unsqueeze(1))
            
            fake = netG(x, y, r, seed)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()
            del G
            
        # Write logs and save samples
        lib.plot.plot('tmp\\mnist3\\time', time.time() - start_time)
        lib.plot.plot('tmp\\mnist3\\train disc cost', D_cost.cpu().data.numpy())
        lib.plot.plot('tmp\\mnist3\\train gen cost', G_cost.cpu().data.numpy())
        lib.plot.plot('tmp\\mnist3\\wasserstein distance', Wasserstein_D.cpu().data.numpy())
    
        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                imgs = torch.Tensor(images)
                if use_cuda:
                    imgs = imgs.cuda(gpu)
            #######################################3
                with torch.no_grad():
                    imgs_v = imgs#, volatile=True)
    
                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('tmp\\mnist3\\dev disc cost', np.mean(dev_disc_costs))
    
            generate_image(iteration, netG)
        
        if iteration % 1000 == 0:
            torch.save(netG.state_dict(), "tmp\\mnist3\\G-cppn-wgan_{}.pth".format(iteration))
            torch.save(netD.state_dict(), "tmp\\mnist3\\D-cppn-wgan_{}.pth".format(iteration))
    
        # Write logs every 100 iters
        if (iteration % 20 == 19):
            lib.plot.flush()
    
        lib.plot.tick()
