# Feature Generation

The purpose of this project is to learn a good neural network architecture via feature search. The `feature_search` was used to figure out how to evaluate features of a fixed structure. This part of the project now extends that to consider different structures. The goal is to find the simplest online feature search algorithm that can outperform a fixed structure MLP given equal resources.


### Plan
- [ ] Setup the problem, which to start will be the standard CIFAR-10 dataset.
- [ ] Implement a training script with a fixed structure MLP and a config.
- [ ] And a sweep config for baseline MLP on CIFAR-10. Include different sizes to know what performance is possible at different sizes.
- [ ] Implement a training script with a feature search algorithm.
