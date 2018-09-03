#coding=utf-8
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import imageio
import model

## samples directions dependent of user choices
class DirectionSampler:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def update_constrain(prediction):
        """used to change the sampler

        Args:
        predictions - list of tuples (,direction) direction is d-dimensional
                      vector indicating direction             
        """
        confidence , directions = prediction
        pass
    
    def generate(number_of_samples):
        return np.zeros(number_of_samples,dimensions)

class StartPositionSampler:
    def __init__(self):
        pass

    def generate(directions, significance_radius, size):
        """samples starting positions of visualization of directions based on
        d-dimensional directions.
        
        Args:
        directions - array of d-dimensional vectors N x D
        significance_radius -
        size (M) - number of starting positions to generate

        returns:
        vector of size M x N x D or M x 1 x D if positions are shared across
        diferent directions. 
        """
        pass

class FramePositionsGenerator:
    def __init__(self):
        pass

    def generate(self, directions, initial_positions, significance_radius,
                 steps = 15):
        """calculates positions during visualization.

        Args:
        directions - array of d-dimensional vectors N x D
        initial_positions - array of d-dimensional vectors M x N x D
                            or M x 1 x D (if there is sharing across directions)
                            from these positions the animation starts
        significance_radius - number denoting how far from distribution
                              should the animation reach
        steps - number of frames during movement

        returns:
        list of positions which are M x N x D vectors
        ( M positions per N directions with D dimensions)
        """
        pass

##implementation of base classes

class Gaussian(DirectionSampler):
    """Samples directions based of normal distribution after linear
    transformations. In not chosen directions the distribution is squeezed
    in chosen is streached

    Args:
    dimensions - number of dimensions of returned and taken vectors
    wrong_coefficient - how much distribution is streached in wrong direction(<1)
    corret_coefficient - how much distribution is streached in desired direction
                        (>=1) use 1.0 if you do not want to discriminate
                        not seen directions
    """
    def __init__(self, dimensions, wrong_coefficient=0.8, correct_coefficient=1.0):
        
        DirectionSampler.__init__(self, dimensions)
        self.linear = np.identity(dimensions)
        self.alpha = correct_coefficient-1
        self.beta = wrong_coefficient-1
        
    def update(self, predictions):
        """

        Args:
        prediction - list of tuples (choice, direction) choice is a number
                     indicating how much correct given direction is;
                     direction is a normalized d-dimensional vector
        """
        for r,x in predictions:
            projection = np.matmul(np.expand_dims(x, axis = 1),
                                   np.expand_dims(x, axis = 0))
            if r>=0:  #correct
                self.linear = np.matmul(self.alpha*projection \
                                        + np.identity(self.dimensions),
                                        self.linear)
            if r<0:   #incorrect
                self.linear = np.matmul(self.beta*projection \
                                        + np.identity(self.dimensions),
                                        self.linear)
                
    def generate(self, number_of_samples):
        """samples d-dimensional normalized vectors based on choices of user.
        """
        samples = np.random.randn(number_of_samples, 1, self.dimensions)
        samples = np.matmul(samples,self.linear).squeeze(1)
        return samples/np.linalg.norm(samples,axis=1,keepdims=True)

    def save(self,file_name):
        np.save(file_name,self.linear)

    def load(self,file_name):
        self.linear = np.load(file_name)

def extend_basis(old_basis):
    #extends basis to make a d-dimensional basis and returns the extension
    current_basis = old_basis
    dimensions = old_basis.shape[-1]
    for i in range(old_basis.shape[0], dimensions):
        x = np.random.randn(dimensions, 1)
        temp = np.linalg.inv(current_basis @ current_basis.transpose())
        projection = current_basis.transpose() @ temp @ current_basis
        x -= np.matmul(projection, x)
        current_basis = np.concatenate([current_basis, x.transpose()])
    return current_basis[old_basis.shape[-2]:,:]


class EqualProjection(StartPositionSampler):
    """Finds positions, which projections to directions are equal and
    multiple of scale parameter divided by normalization constant.
    """
    def __init__(self, dimensions):
        StartPositionSampler.__init__(self)

    def lattice_step(self, directions):
        n, d = directions.shape#TODO: check if directions are linary idependent
        matrix = np.matmul(directions, directions.transpose()) #[n x n]
        result = np.ones(n)
        #projection on directions has to be 1 in other directions equal to 0
        return np.matmul(np.expand_dims(np.linalg.solve(matrix, result),0),
                         directions).squeeze(0)  #TODO:check order of liang solve

    def generate(self, directions, significance_radius, scale, number):
        """samples starting positions of visualization of directions based on
        d-dimensional directions. 
        
        Args:
        directions - array of d-dimensional vectors N x D
        significance_radius - how large varience is
        size (M) - number of starting positions to generate

        returns:
        vector of size M x N x D or M x 1 x D if positions are shared across
        diferent directions. 
        """
        
        step = np.expand_dims(self.lattice_step(directions),axis=0)
        basis = extend_basis(directions)
        d = basis.shape[0] + 1
        coordinates = np.around(np.random.normal(np.zeros(d), np.ones(d),
                                                (number, d))/scale)*scale

        return np.matmul(coordinates, np.concatenate([step, basis],axis=0)) \
               /np.sqrt(d)*significance_radius

class LinearInterpolation(FramePositionsGenerator):
    """animations goes back and forth along direction.
    Interpolation begins at initial positions goes from one edege to other edge
    of sphere of significance radius. The further from center the worse
    images are. 
    """
    def __init__(self):
        FramePositionsGenerator.__init__(self)
        
    def generate(self, directions, initial_positions, significance_radius, steps = 15):
        """calculates positions during visualization.

        Args:
        directions - array of d-dimensional vectors N x D
        initial_positions - array of d-dimensional vectors M x N x D
                            or M x 1 x D (if there is sharing across directions)
                            from these positions the animation starts
        significance_radius - number denoting how far from distribution
                              should the animation reach
        steps - number of frames during movement

        returns:
        list of positions which are M x N x D vectors
        ( M positions per N directions with D dimensions)
        """
        #function descibing movement; has to have period of value 1; x\in [0,1]
        #value has to go from -1 to 1 and back to -1
        #-1 and 1 are edges of significance 
        def function(x): 
            x = np.remainder(x,1)
            return -np.absolute(x-0.5)*4+2
        
        #inverse function has to work in first part of function
        #(in which value goes from -1 to 1; in x's it is from 0 to 0.5 if
        #animation is symetric)
        #to dermine starting x to match with initial position
        def inv_function(x): 
            return (x+1)/4
        
        r = significance_radius
        directions = np.expand_dims(directions,0)
        distance_to_middle = np.sum(initial_positions*directions,axis=2,keepdims=True)
        lenght_of_path = 2*np.sqrt(r**2 - np.linalg.norm(initial_positions \
                                                         -distance_to_middle*directions,
                                                         axis=2,keepdims=True)**2)
        initial_offset = inv_function(-2*np.absolute(distance_to_middle)\
                                      /lenght_of_path)
        result = []
        # np.sin(2*distance_to_middle/lenght_of_path)
        
        for k in range(0, steps-1):
            temp = (function(k/steps + initial_offset) \
                        - function(initial_offset) )*lenght_of_path/2*directions\
                    + initial_positions
            result.append(temp)
        result = [result[0]]*2 +result
        return result        
