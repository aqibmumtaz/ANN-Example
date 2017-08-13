

import numpy as np


# The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with. In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate.

def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video


# The following code creates the input matrix.

X = np.array([[0,0],  # Note: there is a typo on this line in the video
            [0,1],
            [1,0],
            [1,1]])

# The output of the exclusive OR function follows. 

#output data
y = np.array([[0],
             [1],
             [1],
             [0]])


# The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

np.random.seed(1)


# Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.

#synapses
l1MatrixW = 2
l1MatrixH = 3
syn0 = 2*np.random.random((l1MatrixW,l1MatrixH)) - 1

l2MatrixW = l1MatrixH
l2MatrixH = 1
syn1 = 2*np.random.random((l2MatrixW,l2MatrixH)) - 1

print ("======= Neural Network with 1 hidden Layer =======")
print ("\n==== Input Layer nodes ", l1MatrixW, ", Hidden Layer nodes ", l1MatrixH, ", Output Layer nodes ", l2MatrixH, " ====\n")


# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases. 
iterations = 100000
for j in range(iterations):
    
    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print ("\nOutput after training ", iterations, " iterations")
print (l2)
    
    



