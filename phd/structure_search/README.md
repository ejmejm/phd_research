# Feature Generation

The purpose of this project is to learn a good neural network architecture via feature search. The `feature_search` was used to figure out how to evaluate features of a fixed structure. This part of the project now extends that to consider different structures. The goal is to find the simplest online feature search algorithm that can outperform a fixed structure MLP given equal resources.


### What Can Be Searched

There are several dimensions of network structure that can be varied during learning:

- **Connectivity** (implemented): Which units connect to which, how many connections each unit has, and when connections are added or removed. This is the primary focus of this project.
- **Activation function** (implemented): Each hidden unit can use a different activation function, sampled from a set of valid activations at generation time.
- **Weight initialization** (implemented): How new connections and units are initialized when they are created. Currently uses Lecun uniform initialization scaled by fan-in.
- **Objective / update rule** (not yet implemented as a searchable dimension): What loss or auxiliary objective each feature is trained to minimize. This is the most interesting direction long-term, as it connects to the GVF discovery problem, but it is out of scope for this project.

The motivation for learning connectivity in particular is threefold:

1. **Parsimony.** A sparse network that has learned the right structure can match the performance of a dense network with fewer parameters. This matters for resource-constrained settings and for interpretability.
2. **Relieving the experimenter.** Choosing a good architecture is a manual burden. If the network can discover its own connectivity, the experimenter only needs to specify capacity constraints rather than a specific topology.
3. **Structural credit assignment.** When a network knows which inputs are relevant to which outputs, it can adapt faster when the environment changes, because gradient updates flow only through the relevant subnetwork rather than being diluted across all parameters.

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


### Environments

#### Parallel MNIST

The parallel MNIST task concatenates multiple independent MNIST problems into a single input-output pair. Each sub-task uses its own subset of the input vector and its own subset of the output vector, so the ideal network is block-sparse: each output group depends only on the corresponding input group, and the rest of the connections are wasted parameters. This is similar in spirit to how the [Nibbler paper](https://arxiv.org/abs/2311.02215) uses a single environment that under the hood contains multiple instances of the same environment.

This task is a good fit for evaluating connectivity search for two reasons. First, the optimal sparse structure is known, so we have a strong baseline: a block-diagonal network that uses far fewer parameters than a dense network of equivalent width. If the search algorithm recovers something close to this structure, it should match dense performance with a fraction of the parameters. Second, the task supports a non-stationary variant where individual sub-tasks have their labels permuted. A network with the right sparse structure should adapt to changes in a single sub-task faster than a dense network because it does not need to marginalize updates over all sub-tasks.

The task is somewhat contrived—normally if we knew we had completely separate sub-problems we would just split them up. The idea is that this is a controlled proxy for the more general setting where an agent faces a complex problem with many facets, and knowledge is stored in a sparse and distributed way across the network. Learning the right connectivity would then provide a form of structural credit assignment that allows targeted adaptation. Current networks are trained dense, with nearly every neuron contributing to every decision, so this kind of approach may not be generally useful until we move toward sparser representations. But an algorithm that learns sparse structure could itself be part of that transition.


### Plan
- [x] Setup the problem, which to start will be the standard CIFAR-10 dataset.
- [x] Implement a training script with a fixed structure MLP and a config.
- [x] And a sweep config for baseline MLP on CIFAR-10. Include different sizes to know what performance is possible at different sizes.
- [x] Implement the dynamic network.
- [x] Implement a training script with a feature search algorithm.
- [ ] Test different with single activation vs. multiple activation functions.