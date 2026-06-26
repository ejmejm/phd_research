Maybe the problem with IDBD for networks is not the formulation, but that it’s just too nonstationary because of the constant changes to the full network. What happens when you only change one parameter in the network at a time? Or some arbitrary subset of weights? Does the formulation work in that setting? What if you have a bunch of 2 layer network like in Nibbler, where the features of one are used as inputs to another?

I want to try running experiments with non-stationary problems where I know which inputs are noise and which are not, then compare the step-size across weights connected to the two types of inputs. The main metric I will look at across these experiments is the separation between the distributions of step-sizes of distractors and informative feature weights.

The first thing I need to do is establish that IDBD step-sizes actually start to suffer as the problem becomes more non-linear. For that I have two things I want to look at:
- What happens to learned step-sizes of input layer weights as the network gets deeper?
- What happens to the learned step-sizes of input layer weights as the network grows wider?

Those experiments will server as my baseline. If I have a configuration of the network that performs poorly, I can then compare that against a number of alternatives. Specifically I want to look at what happens when:
- Only one weight is ever updated at once.
- Only one layer is ever updates at once.
- Only a random subset of the weights are updated, using different fractions of the total number of parameters.
- Each of the above combined with different types of sparity:
    - Random sparsity
    - Small world sparsity
