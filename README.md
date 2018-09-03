# variant-of-CPPN-GAN
based on https://github.com/kwj2104/CPPN-WGAN, but on chineses fonts and improved architecture


the fonts can be download from https://www.kaggle.com/dylanli/chinesecharacter. You just have to put them in file fonts. To generate images you have to execute script chinese.py by default it will generate 8k 64x64 images. 

## Generator exploration
There are also scipts for more sofisticated image and gif creation
### generator_exploration_main.py
Experimental method of finding sematically meaningfull directions in the latent space. Every iteration it produces an interpolation along 4 directions starting from 9 points in latent space (4*9 interpolations). The user has to decide, which directions show a desired change of property and which do not. Then algorithm will change probabilities of sampling directions. In current implementation the visualization is saved as a gif and the choice of directions is made by typing subset of numbers {0,1,2,3}denoting desired directions (if no direction is wanted press enter). The algorithm is sampling from lineary transformed normal distribution(covariant). Every time the visualized direction is not chosen the distribution is squished along this direction, which make it less probable.  

### image_enhancer.py
Simple scipt to optimize latent space position for better images in cases that current position is during transition between two states. The script just moves point in latent space to closest minimum(with respect to discriminator loss) in the randomly chosen subspace of latent space(the dimensionality of subspace can be chosen)
