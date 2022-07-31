-------------------------------------------------------------------------------------------

### Newton's method

If $x_0$ is near a solution of $f(x)=0$ then we can approximate $f(x)$ by the tangent line at $x_0$ and compute the x intercept of the tangent line. 

We iteratively apply this process using the derived recursive formula.
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

#### Mitigation of non-convergence

 - A large error in the initial estimate can contribute to non-convergence of the algorithm besides good initial estimates lie close to the final globally optimal parameter estimate. 

 - If the derivative is zero the method will terminate due to division by zero. The derivative must be $f'(x_n) \ne 0$.


We place limit on the number of iterations (max_iterations)

Bounding the solution to an interval known to contain the root (using *Intermediate Value Theorem*) can be applied beforehand.

*The intermediate value theorem* (**IVT**) in calculus states that if a function f(x) is continuous over an interval $[a,b]$, then the function takes on every value between $f(a)$ and $f(b)$, then if we can find two endpoints that are of opposite signs that means it must be a x intercept and then at least one solution on that interval

If no bounding is used then we do not know how close we are to a solution, newton's method exhibits one sided convergence in the limit, deviation from this behavior indicates that you are pushing against the limitations imposed by floating point arithmetic. We can then (*this is not a recomended practice, **look up error bounds***) terminate the process using a predefined threshold $\epsilon$ or $\delta$

$|f(x_n)| < \epsilon$ or $|x_n - x_{n-1}| < \delta$ are frequently used as stoppers.

<br/>
The first one check for convergence of the $f(x_n)$ absolute value, $\epsilon$ being the threshold (0.1, 0.0001,1e-12) when the absolute value obtained from applying $x_n$ to $f()$ approaches 0 (root or x intercept) it ceases when it is less than $\epsilon$.


The second check for convergence of the difference between $x_n$ and $x_{n-1}$ absolute values, $\delta$ being the threshold (0.1, 0.0001,1e-12) when this difference is less than $\delta$ it ceases, it is approaching the value of x such that $f(x)=0$
<br/>

We will use $|f(x_n)| < \epsilon$ in this case.

``` python

def newtons_method(f,df,x0,epsilon,max_iterations):
    #initialization of xn to guess
    xn = x0
    dfxn = -1

    for n in range(0,max_iterations):
        #plug xn value into f function we want to find a root 
        fxn = f(xn)
        #print('Value of f(x',n,'):',fxn,'||| value of f\'(x',n,'):',dfxn,'||| absolute f(x',n,'):',abs(fxn),'||| x',n,' value:',xn)
        
        #if the absolute value of applying xn to the function falls under predefined threshold epsilon
        #the desired behaviour is that it stop iterating because it is converging to the root
        #(xn approaching the desired x intercept)
        
        if abs(fxn) < epsilon:
            return xn
        
        #plug xn value into f' (f function derivative)
        dfxn = df(xn)
        
        #we check for stationary point
        if dfxn == 0:
            return float('nan')
        
        #use the derived recursive formula to compute n_x+1 value
        xn = xn - fxn/dfxn
        
    #maximum number of iterations exceeded
    return float('nan')

#--------->

funct = lambda x: x**5 - x*5 + 3
d_funct = lambda x: 5*x**4 - 5

##Applying IVT we know a solution exists between interval [1,2]
##f'(1) == 0 so we go with x0 = 2
approx = newtons_method(funct,d_funct,2,1e-6,20)
print(approx)

```

-------------------------------------------------------------------------------------------

### Newton's fractal

``` python

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

#--------->

# Julia Nova fractal for f(z) = z3 − 1
#funct = lambda z: z**3 - 1
#d_funct = lambda z: 3*z**2

# DiegoAlfa7 suggested me this polynomial
#funct = lambda x: ((2*x+6)**(1/3))+5*x**3
#d_funct = lambda x: ((1/3)*(2*x+6)**(-2/3)) + 15*x**2

#blackpenredpen calculus newton's method video plynomial
#funct = lambda x: x**5 - x*5 + 3
#d_funct = lambda x: 5*x**4 - 5

#3Blue1Brown newton’s method video polynomial 
funct = lambda z: z**5 + z**2 - z + 1
d_funct = lambda z: 5*z**4 + 2 * z - 1

newton_fractal(funct, d_funct, n=500)

```
