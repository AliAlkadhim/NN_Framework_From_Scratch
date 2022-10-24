import numpy as np

class tanh(object):
    @staticmethod #this static method just means we dont have to initialize or pass self, we can do whatever without it
    def calc(v):
        return np.tanh(v)