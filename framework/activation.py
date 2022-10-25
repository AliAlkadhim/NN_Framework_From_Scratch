import numpy as np

class tanh(object):
    @staticmethod #this static method just means we dont have to initialize or pass self, we can do whatever without it
    def calc(v):
        return np.tanh(v)
    
    @staticmethod
    def calc_deriv(v):
        #calculate d tanh(v)/dv = 1- tanh^2 (v)
        #this is basically dy/dy' = d f(y') / dy' , where y' is the output before activation
        return 1- np.tanh(v)**2
