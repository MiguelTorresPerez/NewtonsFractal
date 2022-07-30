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
