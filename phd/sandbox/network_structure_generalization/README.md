This file contains a description of the planned experiment along with motivation.

### Motivation

I want to show how the structure of a network impacts how it generalizes. When I say generalization, however, I mean in the continual learning sense, not in a train/test set gap sense. In this case, generalization is just how what the network learns from any given sample affects how it performs in the future. Learning a representation that allows you to adapt to future change faster, for example, would be an example of good generatlization.

### Problem

MNIST and CIFAR-100 datasets, but changed to be made continual learning tasks by doing the split version of the task, where only 2 classes are presented at a time, and which 2 classes changes at a frequency decided by the experimenter (default 1000 steps for MNIST and 200 steps for CIFAR-100). These should be presented as datastreams where a single sample is presented at a time with no notion of epochs. Samples should be sampled with replacement, and there should be another parameter to decide whether samples can be pulled multiple times (default to true), as opposed to stopping iteration after ever class has been used in a binary comparison. When true, it should be possible to resample the same class but in a different pair of classes.

### Experimental Setup

The general structure of the experiments will be comparing how different model structures and step-size adaptation schemes affect how a model generalizes. The experiments are designed carefully so that just looking at the average performance will tell us how the model is generalizing. The implementation will start with just structural experiments where we look at the difference between how MLPs and CNNs learn.

#### Structural Experiments

1. We construct an MLP and CNN, train them on both datasets, and measure the average performance of each. The models should both have about the same number of parameters (~50k) and the CNN should *not* use weight sharing to make the comparison fair. Both models will be trained with RMSProp. We first need to sweep over the learning rate for both models on both environments and find a good value. Then we will report results on how fast both models learn (average loss), and what their asymptotic performance is (average loss over last 10% of steps when trained sufficiently long). The question is if one model will be better than another when they are constrained to the same resource limits.


Note for later: I could show the connectivity thing by making a network where sparse connectivity leads to less forgetting.