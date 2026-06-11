"""Continual Backprop (CBP) + LMS on the feature-sifting task -- a minimal numpy demo.

The feature-sifting task is an online regression problem. A fixed target function
produces each label as a weighted sum (+/-1 weights) of `n_target_features` true
features drawn uniformly from [-1, 1]. The learner never sees the true features; it
observes `n_learner_features` *candidate* features, each a noisy view of one randomly
chosen true feature (the true value mixed with pure noise by a per-candidate noise
coefficient in [0, 1]). The learner predicts the label with a linear model and, over
time, must *sift* the useful low-noise candidates from the useless ones.

  * LMS = plain online linear regression: an SGD step on the squared prediction error.
  * Continual Backprop adds feature recycling on top: it tracks each candidate's
    "utility" (a running average of |weight| * |feature|) and periodically discards the
    lowest-utility candidate, asking the task to replace it with a freshly drawn one.
    Useful fresh candidates get kept; useless ones get recycled again later.

Only requires numpy and matplotlib. Run `python cbp_lms_demo.py --help` for the
configurable arguments.
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save a plot without needing a display
import matplotlib.pyplot as plt

from feature_sifting import FeatureSiftingTaskNP


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-target-features", type=int, default=10)
    parser.add_argument("--n-learner-features", type=int, default=20)
    parser.add_argument("--target-noise-std", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="LMS (SGD) step size")
    parser.add_argument("--replace-rate", type=float, default=1e-3,
                        help="CBP fraction of features recycled per step")
    parser.add_argument("--decay-rate", type=float, default=0.99,
                        help="CBP utility EMA decay rate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=5000)
    parser.add_argument("--plot-path", type=str, default="cbp_lms_learning_curve.png")
    return parser.parse_args()


def init(args):
    """Set up the problem (task) and the learner (a linear model + CBP bookkeeping)."""
    task = FeatureSiftingTaskNP(args.n_target_features, args.n_learner_features,
                                args.target_noise_std, args.seed)

    n = args.n_learner_features
    w = np.zeros(n)                       # linear weights, one per candidate feature
    util = np.zeros(n)                    # running utility estimate per candidate
    accum = 0.0                           # CBP replacement budget accumulator
    prune_mask = np.zeros(n, dtype=bool)  # which slots to recycle on the next step
    return task, w, util, accum, prune_mask


def train(args, task, w, util, accum, prune_mask):
    """Run the training loop, printing progress and saving the learning curve."""
    n = args.n_learner_features
    losses = np.zeros(args.num_steps)

    print(f"Running CBP + LMS for {args.num_steps} steps "
          f"(lr={args.learning_rate}, replace_rate={args.replace_rate}, "
          f"decay_rate={args.decay_rate})\n")

    for t in range(args.num_steps):
        # The task replaces the slots we pruned last step, then hands us a sample.
        x, y = task.step(prune_mask)

        # LMS: predict, then take an SGD step on the squared error.
        y_hat = np.dot(w, x)
        error = y_hat - y
        losses[t] = error ** 2
        grad = 2.0 * error * x
        w -= args.learning_rate * grad

        # CBP: update the utility estimate (EMA of |weight| * |feature|).
        step_util = np.abs(w) * np.abs(x)
        util = (1 - args.decay_rate) * step_util + args.decay_rate * util

        # CBP: accumulate the replacement budget and recycle the least-useful candidate(s).
        accum += args.replace_rate * n
        n_to_prune = int(np.floor(accum))
        prune_mask = np.zeros(n, dtype=bool)
        if n_to_prune > 0:
            worst = np.argsort(util)[:n_to_prune]  # lowest-utility candidates
            prune_mask[worst] = True
            accum -= n_to_prune
            w[worst] = 0.0                # reset the recycled weights
            util[worst] = np.median(util)  # protect fresh slots from instant re-pruning

        # Occasional progress print.
        if (t + 1) % args.print_every == 0:
            recent = losses[t + 1 - args.print_every: t + 1].mean()
            print(f"step {t + 1:>7d} / {args.num_steps}   recent avg loss = {recent:.4f}")

    final_loss = losses[int(0.95 * args.num_steps):].mean()
    print(f"\nFinal avg loss (last 5% of steps): {final_loss:.4f}")
    return losses


def plot_learning_curve(losses, num_steps, plot_path):
    """Bin the noisy per-step loss into ~250 points and save the learning curve."""
    n_bins = min(250, num_steps)
    bin_size = num_steps // n_bins
    binned = losses[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    bin_steps = np.linspace(0, num_steps, n_bins)

    plt.figure(figsize=(7, 4))
    plt.plot(bin_steps, binned)
    plt.xlabel("step")
    plt.ylabel("squared error (binned mean)")
    plt.title("CBP + LMS on feature sifting")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Saved learning curve to {plot_path}")


def main():
    args = parse_args()
    task, w, util, accum, prune_mask = init(args)
    losses = train(args, task, w, util, accum, prune_mask)
    plot_learning_curve(losses, args.num_steps, args.plot_path)


if __name__ == "__main__":
    main()
