"""
Signed Utility Traces: IDBD vs LMS on a Non-Stationary Tracking Task
=====================================================================
Motivation: Investigate whether "signed utility" traces (measuring how much each input
reduces prediction error) can distinguish relevant from irrelevant features in a
non-stationary online learning setting. This is a precursor to using utility-based
metrics for feature selection or pruning in neural networks.

What it does: Generates a 20-input tracking task where only 5 inputs are relevant,
and the sign of one relevant input flips every 20 steps. Runs two algorithms:
  1. IDBD (Incremental Delta-Bar-Delta): adapts per-feature learning rates online
  2. LMS (Least Mean Squares): fixed learning rate baseline
Tracks three diagnostic traces per input via EMA (decay=0.999):
  - Signed utility: |error_without_feature| - |error_with_feature|
  - Input-target correlation: x[i] * y_star
  - Contribution: |x[i] * w[i]|

Results: IDBD converges to lower asymptotic MSE than LMS. The signed utility and
contribution traces clearly separate relevant inputs (positive, larger magnitude)
from irrelevant ones (near zero), but the signed utility traces immediately go negative,
whereas the contribution traces remain positive.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters
n_inputs = 20
n_relevant = 5
n_examples = 30000
n_train = 20000
n_test = 10000
drift_frequency = 20
trace_decay = 0.999

# IDBD parameters
theta = 0.005
alpha_init = 0.05

# LMS parameters
lms_alpha = 0.03

# Generate tracking task data
np.random.seed(42)
X = np.random.randn(n_examples, n_inputs)
signs = np.ones(n_relevant)
y_target = np.zeros(n_examples)

for t in range(n_examples):
    if t > 0 and t % drift_frequency == 0:
        idx = np.random.randint(n_relevant)
        signs[idx] *= -1
    y_target[t] = np.dot(signs, X[t, :n_relevant])

# ============================================
# Run IDBD Algorithm
# ============================================
w_idbd = np.zeros(n_inputs)
beta = np.log(alpha_init) * np.ones(n_inputs)
h = np.zeros(n_inputs)
errors_idbd = []
alpha_history_idbd = []
utility_traces = np.zeros(n_inputs)
utility_history = np.zeros((n_examples, n_inputs))
correlation_traces = np.zeros(n_inputs)
correlation_history = np.zeros((n_examples, n_inputs))
contribution_traces = np.zeros(n_inputs)
contribution_history = np.zeros((n_examples, n_inputs))

for t in range(n_examples):
    x = X[t]
    y_star = y_target[t]
    
    # Predict
    y = np.dot(w_idbd, x)
    delta = y_star - y
    
    # Compute utility for each input
    error_with = np.abs(delta)
    for i in range(n_inputs):
        error_without = np.abs(delta + x[i] * w_idbd[i])
        utility = error_without - error_with  # Positive if input helps
        utility_traces[i] = utility_traces[i] * trace_decay + utility * (1 - trace_decay)
    
    utility_history[t] = utility_traces.copy()
    
    # Compute correlation trace for each input (x[i] * y_star)
    correlation_traces = correlation_traces * trace_decay + x * y_star * (1 - trace_decay)
    correlation_history[t] = correlation_traces.copy()

    # Compute contribution utility trace: EMA of |x[i] * w[i]|
    contribution_traces = (
        contribution_traces * trace_decay
        + np.abs(x * w_idbd) * (1 - trace_decay)
    )
    contribution_history[t] = contribution_traces.copy()

    # Update beta (log learning rates)
    beta += theta * delta * x * h
    
    # Compute learning rates
    alpha = np.exp(beta)
    
    # Update weights
    w_idbd += alpha * delta * x
    
    # Update traces with positive bounding
    h = np.maximum(0, h * (1 - alpha * x**2)) + alpha * delta * x
    
    errors_idbd.append(delta**2)
    if t % 100 == 0:
        alpha_history_idbd.append(alpha.copy())

errors_idbd = np.array(errors_idbd)
alpha_history_idbd = np.array(alpha_history_idbd)

# ============================================
# Run LMS Algorithm
# ============================================
w_lms = np.zeros(n_inputs)
errors_lms = []

for t in range(n_examples):
    x = X[t]
    y_star = y_target[t]
    
    # Predict
    y = np.dot(w_lms, x)
    delta = y_star - y
    
    # Update weights
    w_lms += lms_alpha * delta * x
    
    errors_lms.append(delta**2)

errors_lms = np.array(errors_lms)

# ============================================
# Print Results
# ============================================
asymptotic_idbd = np.mean(errors_idbd[n_train:])
asymptotic_lms = np.mean(errors_lms[n_train:])

print("Running IDBD vs LMS comparison...")
print("=" * 50)
print(f"Asymptotic Mean Squared Error (last {n_test} examples):")
print(f"  IDBD: {asymptotic_idbd:.3f}")
print(f"  LMS:  {asymptotic_lms:.3f}")
print(f"  Improvement: {(1 - asymptotic_idbd/asymptotic_lms)*100:.1f}%")
print()

final_alphas = np.exp(beta)
print("Final IDBD learning rates:")
print(f"  Relevant inputs (1-5):   {final_alphas[:5]}")
print(f"  Irrelevant inputs (6-10): {final_alphas[5:10]}")
print(f"  Mean (relevant):   {np.mean(final_alphas[:5]):.4f}")
print(f"  Mean (irrelevant): {np.mean(final_alphas[5:]):.4f}")

# ============================================
# Plot Results
# ============================================
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 13))

# Plot 1: Error curves
window = 500
errors_idbd_smooth = np.convolve(errors_idbd, np.ones(window)/window, mode='valid')
errors_lms_smooth = np.convolve(errors_lms, np.ones(window)/window, mode='valid')

ax1.plot(errors_idbd_smooth, label='IDBD', linewidth=1.5)
ax1.plot(errors_lms_smooth, label='LMS', linewidth=1.5)
ax1.set_xlabel('Examples')
ax1.set_ylabel('Mean Squared Error (smoothed)')
ax1.set_title('Tracking Performance: IDBD vs LMS')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Step-sizes over time
time_steps = np.arange(0, n_examples, 100) / 1000
ax2.plot(time_steps, alpha_history_idbd[:, 0], label='Relevant input', linewidth=2)
ax2.plot(time_steps, alpha_history_idbd[:, 10], label='Irrelevant input', linewidth=2)
ax2.set_xlabel('Time Steps (1000s of Examples)')
ax2.set_ylabel('Step-size (α)')
ax2.set_title('Evolution of Step-sizes under IDBD')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Utility traces over time
time_axis = np.arange(n_examples)
for i in range(n_inputs):
    if i < n_relevant:
        # Relevant inputs: blue
        ax3.plot(time_axis, utility_history[:, i], color='blue', alpha=0.3, linewidth=1)
    else:
        # Irrelevant inputs: red
        ax3.plot(time_axis, utility_history[:, i], color='red', alpha=0.2, linewidth=0.8)

# Add dummy lines for legend
ax3.plot([], [], color='blue', alpha=0.6, linewidth=2, label='Relevant inputs (1-5)')
ax3.plot([], [], color='red', alpha=0.6, linewidth=2, label='Irrelevant inputs (6-20)')

ax3.set_xlabel('Examples')
ax3.set_ylabel('Signed Utility Trace')
ax3.set_title('Evolution of Signed Utility Traces (decay=0.01)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Plot 4: Contribution utility traces over time (EMA of |x[i] * w[i]|)
for i in range(n_inputs):
    if i < n_relevant:
        ax4.plot(time_axis, contribution_history[:, i], color='blue', alpha=0.3, linewidth=1)
    else:
        ax4.plot(time_axis, contribution_history[:, i], color='red', alpha=0.2, linewidth=0.8)

ax4.plot([], [], color='blue', alpha=0.6, linewidth=2, label='Relevant inputs (1-5)')
ax4.plot([], [], color='red', alpha=0.6, linewidth=2, label='Irrelevant inputs (6-20)')
ax4.set_xlabel('Examples')
ax4.set_ylabel('Contribution Utility Trace')
ax4.set_title('Evolution of Contribution Utility Traces (decay=0.01)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

# Plot 5: Absolute Correlation traces over time (commented out for now)
# for i in range(n_inputs):
#     if i < n_relevant:
#         # Relevant inputs: blue
#         ax5.plot(time_axis, np.abs(correlation_history[:, i]), color='blue', alpha=0.3, linewidth=1)
#     else:
#         # Irrelevant inputs: red
#         ax5.plot(time_axis, np.abs(correlation_history[:, i]), color='red', alpha=0.2, linewidth=0.8)
#
# # Add dummy lines for legend
# ax5.plot([], [], color='blue', alpha=0.6, linewidth=2, label='Relevant inputs (1-5)')
# ax5.plot([], [], color='red', alpha=0.6, linewidth=2, label='Irrelevant inputs (6-20)')
#
# ax5.set_xlabel('Examples')
# ax5.set_ylabel('Absolute Correlation Trace')
# ax5.set_title('Evolution of Absolute Input-Target Correlation Traces (decay=0.01)')
# ax5.legend()
# ax5.grid(True, alpha=0.3)
# ax5.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('signed_utility_results.png', dpi=150)