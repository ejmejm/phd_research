# Continual Atari Prediction

The goal of this project is to see if current continual learning algorithms (mainly UPGD) will generalize across Atari games. The way I will test this is by comparing the average reward of my proposed method to two baselines. The first baseline is a method with a network that resets for every game. A comparison with this will tell me if the knowledge from prior games is helping or not. The second baseline is a network with no resets trained with Adam. Comparing to this baseline will tell me if the continual learning additions are doing anything to help over the standard way of doing things.

### TODO

First I need to make some change to the [benchmark](https://github.com/ejmejm/ContinualAtariBenchmark):
- [x] Collect 500k+ samples from all Atari 100k benchmark games
- [x] Update the continual Atari benchmark to work with pre-recorded episodes
- [x] Allow for a batch learning option?

Then I need to get a baselines running:
- [x] Implement a basic multi-seed multi-run logging script with preloading of batches of data, following `phd/sturcture_search/train.py` as an example
- [x] Swap out the data with the Atari prediction dataset (found in `../continual_atari_benchmark/dataset/dataset_loader.py`). Make a symlink to the data in that same dir so it can be used for this project.
- [x] Add a preprocessing step to resize to 84x84, grayscale, and do framestacking of 4
- [x] Implement the ResNet used in the Bigger, Better, Faster paper
- [x] Setup a training script that uses Adam
- [x] Setup a training script that uses Adam, and reinitializes the network at game boundaries

Then I need to setup my proposed solution method:
- [x] Add an option to use an UPGD optimizer

Then I need to prepare sweeps. I want to compare the different methods with a good set of hyperparameters for each method:
- [x] Write a sweep config for Adam, sweeping the step-size, and whether network resets are used
- [x] Write a sweep config UPGD, sweeping all of the things that were swept in the original paper
- [x] Test these experiments to see if they are efficient enough to run on my own computer
- [ ] Setup a CC configuration for running these experiments?
- [ ] Run the CC experiments?

This starts with just using supervised learning to predict precomputed returns, but if that works I will want to move onto trying to learn a value function:
- [ ] Make sure rewards are clipped like they are for computing the returns
- [ ] Start with one step TD for the error function
- [ ] Consider using TD lambda for the error function

Other things to try that in general make learning better:
- [ ] Normalize the observations with a Wellford-style online normalization
- [ ] Use something like SR-SPR to learn better representation and more potentially useful features
- [ ] Try batching to stabalize gradients