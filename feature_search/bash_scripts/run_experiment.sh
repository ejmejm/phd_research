python run_experiment.py --config-name=nonlinear_geoff.yaml \
device=cpu \
seed=0 \
\
task.n_real_features=10 \
task.n_features=200 \
task.weight_scale=0.5 \
task.flip_rate=0.001 \
task.n_layers=1 \
task.hidden_dim=20 \
task.sparsity=0.0 \
\
input_recycling.distractor_chance=0.0 \
input_recycling.recycle_rate=0.0 \
input_recycling.use_cbp_utility=true \
input_recycling.feature_protection_steps=100 \
input_recycling.utility_decay=0.999 \
\
train.optimizer=idbd \
train.batch_size=1 \
train.total_steps=1000000 \
train.log_freq=100 \
train.learning_rate=0.005 \
train.weight_decay=0.0 \
\
model.n_layers=2 \
model.hidden_dim=128 \
model.weight_init_method=zeros \
\
idbd.meta_learning_rate=0.1 \
idbd.version=squared_grads \
\
wandb=true

# I'm having problems with getting any of the IDBD versions to work with a multi-layer model.
# With the squared_grads version, the learning rate of the distractor does at least seem to decrease at a near-linear rate, but it is still very slow.
# Perhaps worse, the weight norm of the distractors increases! Note that it does look like it may stabalize (at around ~1), but it still clearly increases.
# At least it does not increase as much as the weights of the real inputs, but the weights of the real inputs also increase far beyond what is necessary (~8).
# My newest experiment shows that if I at least use strong regularization, then the weight of the distractors will start to decrease eventually, but still,
# the rate at which the step-size of the distractors decreases is inordinately slow.
# There is a really interesting question to answer here, which is why when switching from the linear to non-linear regime does the magnitude of all of
# the weights suddenly shoot up at the beginning of training regardless of whether or not they are useful. 