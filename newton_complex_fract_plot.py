import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def newton_fractal(f, fprime, n=100, domain=(-1, 1, -1, 1)):
    #initialize root array that will have, inferring from the Fundamental theorem of algebra: 
    #A degree n polynomial with complex coefficients has, counted with multiplicity, exactly n complex roots
    roots = []
    #fill a n x n zero square matrix
    m = np.zeros((n, n))

    def get_root_index(roots, r):
        try:
            return np.where(np.isclose(roots, r, atol=1e-8))[0][0]
        except IndexError:
            roots.append(r)
            return len(roots) - 1
          
    #The domain used for the fractal image is the region of the complex plane
    #(xmin, xmax, ymin, ymax) where z = x + iy, discretized into n values along
    #each axis.
    xmin, xmax, ymin, ymax = domain
    
    #each coordinate of the domain of dimension n along x and y axis (all n values) is used as z0 (initial guess)
    #to compute in which root is going to converge and represent it in the matrix with a color associated to that root
    for ix, x in enumerate(np.linspace(xmin, xmax, n)):
        for iy, y in enumerate(np.linspace(ymin, ymax, n)):
            z0 = x + y*1j
            r = newtons_method(f, fprime,z0,1e-6,50)
            #if newtons_method find a root get its index in roots (with a sensitivity of atol), if it does not exist add it 
            if r is not False:
                ir = get_root_index(roots, r)
                #keep track of the root index each point belong to and save it into m
                m[iy, ix] = ir
    plt.imshow(m, cmap='hsv', origin='lower')
    plt.axis('off')
    plt.show()

funct = lambda z: z**5 + z**2 - z + 1
d_funct = lambda z: 5*z**4 + 2 * z - 1

newton_fractal(funct, d_funct, n=500)
