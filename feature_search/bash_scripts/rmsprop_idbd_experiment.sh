python run_experiment.py --config-name=nonlinear_geoff.yaml \
device=cpu \
seed=0 \
\
task.n_real_features=10 \
task.n_features=20 \
task.weight_scale=0.5 \
task.flip_rate=0 \
task.n_layers=2 \
task.hidden_dim=20 \
task.sparsity=0.0 \
\
input_recycling.distractor_chance=0.0 \
input_recycling.recycle_rate=0.0 \
input_recycling.use_cbp_utility=true \
input_recycling.feature_protection_steps=100 \
input_recycling.utility_decay=0.999 \
\
train.optimizer=rmsprop_idbd \
train.batch_size=1 \
train.total_steps=50000 \
train.log_freq=100 \
train.learning_rate=0.002 \
\
model.n_layers=2 \
model.hidden_dim=20 \
model.weight_init_method=zeros \
\
idbd.meta_learning_rate=0.002 \
idbd.diagonal_approx=true \
\
wandb=true


# meta step-size=5, batch size=16, learning rate=0.002, diagonal approx=true, no momentum works well
# Findings so far:
# - learning rate of 0.002 works best for batch size of 1 and 16
# - Most hypers can easily get to ~0.05 loss. Good hypers with 1 batch size can get to <0.01 loss in 50k steps, and < 0.001 loss with 16 batch size
# - When using a batch size of 16, the meta learning rate can go up to 50 and work perfectly fine. With a batch size of 1, the meta learning rate cannot go nearly as high.
#   This smoothing effect of greater batch sizes is something that is known, but I should keep it in mind when trying to get the best results in the batch size of 1 case.
# - With a 2 layer network with 20 hidden units and no distractors, both using the hessian diagonal and hessian vector product work fine (but I haven't tested them in
#   a case with distractors yet, and that is where this really should be tested).
# - When I add distractors, RMSProp alone cannot do nearly as well, as expected. The loss gets stuck around 0.06.
# - When I increase the meta learning rate to above 0 in the case with distractors, the model is able to do a little better, getting to around 0.05 loss, but it's still bad.
#   With a batch size of 16, the learning rate of the real features rises high rapidly, then stabilizes.
#   With a batch size of 1, the learning rate of the real features raises a little at the beginning of training, then stabilizes.
#   In both cases, the learning rate of the distractors only drops a negligible amount before stabilizing. This needs to be solved, but I don't know why it's happening.
#   It could be due to the (intentionally) incorrect IDBD h update equation, a way IDBD inherently works, a bug, or something else.
# - Going back, even in the linear case a similar problem persists. It takes a lot longer for a learning rate to half than it does to double. I imagine this is because
#   h is generally going to be smaller in distractors where the gradient gets pulled randomly in different directions.
#   This problem is more pronounced in the non-linear case, but it is not only present there. In the linear case at least, the learning rate of the distractors decreases
#   at a near linear rate, with only a slight curve towards stabilization. My first thought regarding the problem of the learning rate decreasing too slowly was to
#   simply increase the meta learning rate, but before I get to a satisfactory learning pace, the high meta learning rate will cause runs to diverge.


# Experiment planning notes:
# - When I run experiments, I think I should compare 3 methods: HVP, hessian diagonal, and squared inputs for the h update.