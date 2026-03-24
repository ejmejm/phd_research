# Learning by Pruning

In my structure search project, I've recently been working on trying to find an algorithm that can find a better network structure via a continual process of generation and pruning, but almost everything I have trying has been not working. The goal of this experiment is to strip out all complexity and attempt that problem in its simplest form.

### The Plan

The plan is to start with a 2 layer MNIST network (1 hidden layer), show the network a single sample at a time, keep a trace of each hidden unit's signed utility, and prune the hidden unit with the lowest utility every fixed number of steps. The question is whether this process alone can result in significantly better than random performance before all of the hidden units are pruned. I will plot the accuracy of the network at each time step to see how it progresses.

If that works, then I will move onto 3 layers and repeat the process. For 3 layers I still need to work out the utility function I will use, but I will use one of the variants of signed utility that I have been working on in my signed utility extension project.