

import numpy as np



# Print floats in readable format to print like float: 3.0, or float: 12.6666666666.
np.set_printoptions(formatter={'float': lambda x: 'float: ' + str(x)})


# This code is a definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with.

def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


# The following code creates the input matrix, our network consists of two input nodes and one output node

X = np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])

# The output of the exclusive OR function as follows.

y = np.array([[0],
             [1],
             [1],
             [0]])


# The seed for the random generator to return the same random numbers each time for being deterministic, which is very useful for debugging.
np.random.seed(1)


# Initialization of weights to random numbers. syn0 is weight matrix between input layer and first hidden layer.

# Synapses
l0Nodes = 2
l1Nodes = 3
syn0 = 2*np.random.random((l0Nodes,l1Nodes)) - 1


l2Nodes = 1
syn1 = 2*np.random.random((l1Nodes,l2Nodes)) - 1

print("\n======= Neural Network with (with one hidden layers) =======")
print("==== Network Topology ", l0Nodes, " x ", l1Nodes, " x ", l2Nodes, " ====\n")


# This is iteration training loop for network training. error decreases on each cycle of training by the slop of sigmoid function using gradient descent and back propagation.
iterations = 100000
for j in range(iterations):
    
    # Calculating forward through out the network
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Calculating error
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    # Back propagation of errors using the chain rule.
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    # Updating weights (no alpha learning term here..)
    # Default Weight assignment equation : W = W + alpha.input.error
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("\nOutput after training ", iterations, " iterations")
print(l2)
    
    



