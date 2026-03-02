# Feature Generation

The purpose of this project is to learn a good neural network architecture via feature search. The `feature_search` was used to figure out how to evaluate features of a fixed structure. This part of the project now extends that to consider different structures. The goal is to find the simplest online feature search algorithm that can outperform a fixed structure MLP given equal resources.


### Structure Search Algorithm

This is the first naive version of the structure search algorithm.

The user provides parameters including:
- Maximum number of parameters
- Maximum number of hidden units per layer
- Maximum number of layers
- Min/max connections per hidden unit
- Valid activations (allows for different units to use different activations)

The model starts as a linear model with no hidden layers, and each input connected to all outputs.
Each step:
- Do one training step on weights and step-sizes.
- Call the structure manager:
    - Update utility statistics for each connection.
    - Prune the least useful p% of connections.
    - Prune any hidden units with no outgoing connections.
    - Compute the number of available connections (total capacity - current connection count).
    - Create remaining (capacity / average parameters per hidden unit) hidden units.
        - Randomly sample an activation function, and use a vector of indices to represent which activation function is used for each hidden unit.
        - Randomly sample inputs from the min/max connections per hidden unit.
        - Randomly choose inputs as any input unit or hidden unit not in the final hidden layer.
        - Set input weights to small random values (lecun uniform using fan in).
        - Set incoming step-size to the a user set param (default to 1e-15).
        - Connect to outputs with initial 0 values.
        - Set outgoing step-size default to mean of outgoing weight step-sizes.
        - Initialize utility to median utility of all connections.


### Dynamic Network Implementation

Having a dynamic network structure is tricky in JAX because it needs to be done with constant sized shapes to avoid recompilation. To make this work, standard neural network layers are insufficient. Instead this needs to be implemented with all memory preallocated to the maximum possible size for weights and associated metadata, then padding can be used to handle the variable number of hidden units and connections. At initialization the network should preallocate a weight matrix for each layer, which is a matrix of shapes (maximum connections per hidden unit, maximum number of hidden units per layer). Then each layer also holds a set of input indices for each shape. Unused input indices are padded with -1, and a separate vector is used as a padding mask to indicate which hidden units are being used. Keeping the weight matrices in a 2D matrix like this should allow for easier reuse of the existing optimizers and other code I was using.

At the start of a forward pass, a vector of size equal to the total number of inputs, hidden units, and outputs is created, and the buffer is used as a store of all activation/input values. This is needed because each layer can take inputs from any prior layer. The forward function scans through each layer, gathers the inputs, and then performs the matrix multiplication before applying the activation. Every valid activation should be applied, but only a single value chosen with a where statement. I'm not sure if it is possible to fuse the gather and matrix multiplication for better performance, but do that if JAX has a built in function for it.

All computation in the network should avoid jax.lax.cond and instead use jax.lax.switch to handle the different cases for better performance.


### Plan
- [x] Setup the problem, which to start will be the standard CIFAR-10 dataset.
- [x] Implement a training script with a fixed structure MLP and a config.
- [x] And a sweep config for baseline MLP on CIFAR-10. Include different sizes to know what performance is possible at different sizes.
- [ ] Implement the dynamic network.
- [ ] Implement a training script with a feature search algorithm.
