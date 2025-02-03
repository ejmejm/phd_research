python run_experiment.py --config-name=nonlinear_geoff.yaml \
device=cpu \
seed=0 \
\
task.n_real_features=10 \
task.n_features=10 \
task.weight_scale=0.5 \
task.flip_rate=0 \
task.n_layers=2 \
task.hidden_dim=20 \
task.sparsity=0.8 \
\
input_recycling.distractor_chance=0.0 \
input_recycling.recycle_rate=0.0 \
input_recycling.use_cbp_utility=true \
input_recycling.feature_protection_steps=100 \
input_recycling.utility_decay=0.999 \
\
train.optimizer=rmsprop \
train.batch_size=16 \
train.total_steps=50000 \
train.log_freq=100 \
train.learning_rate=0.002 \
\
model.n_layers=2 \
model.hidden_dim=20 \
model.weight_init_method=zeros \
\
idbd.meta_learning_rate=0.0 \
idbd.diagonal_approx=false \
\
wandb=false