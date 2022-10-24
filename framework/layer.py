import numpy as np

class Dense(object):
    def __init__(self,
    N_inputs,
    N_outputs,
    activation
    ):
        """Define the number of input and output NODES for a particular layer. A layer could then be made by calling 
        layer= Dense(N_inputs,N_outputs). For example, a single layer NN with no hidden layers could be done with
        model_1=[Dense(N_inputs,N_outputs)]
        all these things are first initialized randomly here, and then they pass to the later functions of this class
        """
        self.N_inputs = int(N_inputs)
        self.N_outputs = int(N_outputs)
        self.activation=activation
        self.learning_rate=int(1e-1)


        #the size of the weights is a matrix of ( N_inputs + 1) X (N_outputs) (and the inputs) is w[N_inputs]+b which is the rows, and the outputs will have [N_outputs]
        rows = self.N_inputs+1#the +1 because there will be a bias vector 
        columns = self.N_outputs
        self.weights = np.random.sample(size=(rows,columns))
        #random sample returns a random unifrom between0 and 1
        self.w_grad = np.zeros((self.N_inputs+1, self.N_outputs))

        #Define set of inputs coming in to the network
        self.x = np.zeros((1,self.N_inputs+1))
        self.y=np.zeros((1,self.N_outputs))

    def forward_propagate_layer(self, inputs):
        """propagate the inputs forward through the NN

        Args:
            inputs : (INPUT TO THE LAYER) vector of values of size [1,N_input]

        Returns:
            y : of size [1, N_out]
        """
        #inputs are the inputs to the layer
        #make a 1X1 bias matrix of ones
        bias=np.ones((1,1))
        #stack the bias on top of the inputs by adding it as a new column (axis 1)
        self.x = np.concatenate((inputs, bias), axis=1)
        #matrix-multiply the selt of augmented inputs x with the weights
        self.y_intermediate = self.x @ self.weights
        #the shapes of what being multiplied is the following:
        #[1, N_in +1] X [N_in +1, N_out] = [1, N_out]

        #Perform activation on the output for the final output
        self.y = self.activation.calc(self.y_intermediate)
        return self.y
