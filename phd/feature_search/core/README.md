## Available Tasks

Tasks are defined in `phd/feature_search/core/tasks.py` and include:

### `dummy`
Generates random inputs and outputs for testing purposes.

### `linear_geoff`
Implements the bit-flipping regression task from the [original IDBD paper](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf). Features include:
- Some features are distractors, others have ideal weights of -1 or +1
- One useful feature weight flips every 20 steps
- Regression problem structure

### `static_linear_geoff`
Similar to `linear_geoff`, but with static ideal weights (no bit-flipping throughout the task).

### `nonlinear_geoff`
A generalized version of the Geoff task supporting:
- Multi-layer target networks
- Hidden layer activations
- Time-varying target network weights
- Binary or Kaiming uniform weight initialization

This task is demonstrated in [this talk](https://youtu.be/qcdNaVAyeQ4?si=Tc2xkqTTvVJnKu_S) and detailed in [Mahmood & Sutton (2013)](http://incompleteideas.net/papers/MS-AAAIws-2013.pdf).