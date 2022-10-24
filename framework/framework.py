###Framework module
import numpy as np


class ANN(object):
    """

    Args:
        model:  a list of layers with the weights and input/output nodes initialized

    """
    def __init__(self, 
    model, 
    expected_input_range=(0,1)
    ):
        self.layers= model
        #each self.layer is a member of the layers.Dense(N_in,N_out) class, so that you can call layer.forward_prop

        #it helps to set the numbers as integers so that we can call range on it later
        self.n_iter_train = int(1e3)#number of iterations to train
        self.n_iter_evaluate = int(1e2) #number of iterations to evaluate on
        self.expected_input_range=expected_input_range

    def normalize(self, values):
        expected_range=self.expected_input_range
        expected_min, expected_max = expected_range
        scale_factor = expected_max - expected_min
        offset = expected_min
        scaled_values = (values - offset)/scale_factor - 0.5#this -0.5 is there so that the lowest value is -0.5

    def denormalize(self, normalized_values):
        expected_range=self.expected_input_range
        expected_min, expected_max = expected_range
        scale_factor = expected_max - expected_min
        offset = expected_min
        return (normalized_values + 0.5) * scale_factor + offset


    def train(self, training_set):
        """

        Args:
            training_set (GENERATOR): its a generator function for the data, so call it by doing next(training_set())
        """
        for iter in range(self.n_iter_train):
            x = next(training_set()).ravel()#ravel flattens it to 1-d array
            y=self.forward_propagate_data(x)
            print('train y', y)

    def evaluate(self, evaluation_set):
        for iter in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
            y=self.forward_propagate_data(x)
            print('evaluate y', y)
    
    def forward_propagate_data(self, x):
        """Forward propagate the inputs to the entire NN (ie the data)

        The inputs x here are the inputs to the entire NN (the data)
        Since the layers.Dense(inputs, outputs) expects a 2d array for the inputs, we have to make our 1D input of shape (1,) into a 2D array of shape (1,N_inputs)"""
        y = x.ravel()[np.newaxis,:]
        #forward propagate through each layer
        for layer in self.layers:
            y = layer.forward_propagate_layer(y)
        #remember that layer is a layers.Dense member
        return y.ravel()
