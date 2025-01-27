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
feature_recycling.distractor_chance=0.0 \
feature_recycling.recycle_rate=0.0 \
feature_recycling.use_cbp_utility=true \
feature_recycling.feature_protection_steps=100 \
feature_recycling.utility_decay=0.999 \
\
train.optimizer=sgd \
train.batch_size=1 \
train.total_steps=1000000 \
train.log_freq=100 \
train.learning_rate=0.005 \
train.weight_decay=0.0 \
\
model.n_layers=1 \
model.hidden_dim=128 \
model.weight_init_method=zeros \
\
wandb=true