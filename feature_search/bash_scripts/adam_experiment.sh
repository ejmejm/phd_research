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
feature_recycling.distractor_chance=0.0 \
feature_recycling.recycle_rate=0.0 \
feature_recycling.use_cbp_utility=true \
feature_recycling.feature_protection_steps=100 \
feature_recycling.utility_decay=0.999 \
\
train.optimizer=adam \
train.batch_size=16 \
train.total_steps=50000 \
train.log_freq=100 \
train.learning_rate=0.005 \
\
model.n_layers=2 \
model.hidden_dim=128 \
model.weight_init_method=zeros \
\
idbd.meta_learning_rate=0.1 \
\
wandb=false

# batch size of 1 and lr of 0.002 also works. These also both work with no sparsity, but it takes much longer to converge.