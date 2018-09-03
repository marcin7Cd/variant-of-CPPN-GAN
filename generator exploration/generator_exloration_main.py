#coding=utf-8
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import imageio
import model
import position_calculation
digits="0123456789"
result=0
to_take=[]
end_of_number=False

def ask_for_sequence():
    digits="0123456789"
    result=""
    to_take=[]
    end_of_number=False
    print('')
    user_input = input("choose directions")+" "
    for x in user_input:
        if x in digits:
            end_of_number=False
            result+=x
        else:
            if not end_of_number:
                end_of_number=True
                if result!='':
                    to_take.append(int(result))
                result=""
    return to_take

#helper function
def generate_animation(file_name,result,direction_size,noise_size):
    dim_dir_x, dim_dir_y = direction_size
    dim_noise_x, dim_noise_y = noise_size
    movie = []
    i=0
    for x in result:
        if x.shape[0] != dim_noise_x*dim_noise_y:
            print("wrong size of direction")
        if x.shape[1] != dim_dir_x*dim_dir_y:
            print("wrong size of noise")
        x = np.reshape(x,(dim_noise_x,dim_noise_y,
                          dim_dir_x,dim_dir_y,
                          mo.noise_dim))
        x = np.transpose(x,[3,1,2,0,4])
        x = np.reshape(x,(x.shape[0]*x.shape[1],
                          x.shape[2]*x.shape[3],
                          x.shape[4]))
        x =np.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
        print(i, end=' ')
        i += 1
        #x = np.stack(x)
        #print(x.shape)
        image = np.stack(np.split(mo.generate_image(x),dim_dir_x*dim_noise_x))
        image = np.transpose(image,[1,3,0,4,2])
        image = np.reshape(image,(image.shape[0]*image.shape[1],
                                  image.shape[2]*image.shape[3],image.shape[4]))
        image *= 255
        image = image.astype('uint8')
        movie.append(image)
    imageio.mimsave(file_name+'.gif',movie)

DIM_DIR_X=2
DIM_DIR_Y=2
DIM_NOISE_X=4
DIM_NOISE_Y=4
LOADING_DIRECTORY = 'distr.npy'
SAVING_DIRECTORY = 'distr.npy'

NUMBER_OF_DIRECTIONS = DIM_DIR_X*DIM_DIR_Y
NUMBER_OF_INITIALS = DIM_NOISE_X*DIM_NOISE_Y
    
if __name__ == "__main__":

    #creating model
    mo = model.Model(NUMBER_OF_DIRECTIONS*NUMBER_OF_INITIALS,
                    "G-cppn-wgan_158000.pth")

    #creating direction sampler
    gen = position_calculation.Gaussian(mo.noise_dim,
                                        #<1 smaller makes not chosen direction less likely
                                        wrong_coefficient=0.7,
                                        #>=1 bigger makes chosen direction more likely
                                        #but makes not seen directions less likely
                                        correct_coefficient=1.0)

    #creating initial position sampler
    lattice = position_calculation.EqualProjection(mo.noise_dim)

    #creating frame position generator
    camera = position_calculation.LinearInterpolation()

    ###loading direction generator
    if os.path.isfile(LOADING_DIRECTORY):
        gen.load(LOADING_DIRECTORY)
    
    while True:
        directions = gen.generate(NUMBER_OF_DIRECTIONS)
        result = camera.generate(directions,
                                 initial_positions = np.expand_dims(lattice.generate(directions, 1, 0.1, NUMBER_OF_INITIALS),axis=1),
                                 significance_radius = np.sqrt(mo.noise_dim)*1.5,
                                 steps = 40)
        
        generate_animation('direction_visualizations', result,
                           (DIM_DIR_X, DIM_DIR_Y), (DIM_NOISE_X, DIM_NOISE_Y))

        ###asking user for choice of directions
        to_accept = ask_for_sequence()
        to_accept = list(map(lambda x: x % NUMBER_OF_DIRECTIONS,to_accept))
        print("taken directions",to_accept)
        prediction = []
        for i in range(NUMBER_OF_DIRECTIONS):
            if i in to_accept:
                prediction.append(( 1,directions[i]))
            else:
                prediction.append((-1,directions[i]))
        gen.update(prediction)
        
        ###saving direction generator
        gen.save(SAVING_DIRECTORY)
        
