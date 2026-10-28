# Incremental generalization

## Latest result: small-network loss plateaus with the 01 update

The shared pool's absent-task RMSE after readout refitting is 1.006 initially, 1.047 after 0.5M examples, 1.049 after 1M and 1.049 after 4M. Most best-matching features are separate across tasks. The fixed old readout also plateaus, around 1.12. Five paired seeds; the 3.5M-example continuation took 95 seconds on GPU.

This is the acceptable small settling loss seen in 01, over the tested horizon. Earlier claims that the small network demonstrated a harmful ongoing retention problem were too strong. The earlier long-run deterioration used the changed error-trace rule. Preserve 01's own-weight update and investigate the additional effect of changing lower features next; indefinite stability is not yet established.

## Current conclusion after reconciling with 01

Separate readouts are not the cause: a shared readout also forgets. The small network's learned upper features interfere during further fitting. With the original 01 update, giving each task fifteen of the same thirty upper units lets both learn almost perfectly and preserves the absent task (refitted RMSE 1.001 → 1.001). A shared pool gives 1.006 → 1.047. This privileged reference establishes sufficient usable capacity; the difficulty is learning how to allocate it.

**Correction to the earlier backward-rule claim:** replacing per-weight derivative traces with error traces changes online learning, even with one trainable layer. It is not just the missing upstream bug fix. Stored error traces can move weights when the current hidden gradient is zero. Using 01's update from initialization improves acquisition and retention; switching only after acquisition missed that effect. Preserve 01's own-weight update when investigating the missing upstream path.

The notebook contains the matched readout/input/rule comparisons, the executable allocation reference, and a 30-case zero-current-gradient check. Earlier sections are the investigation history; this is the current interpretation.

**Question 1 is the priority:** with ample capacity, why would learning one task damage a previously useful computation, and why would Autostep allow it? Reuse and learning variations are deferred.

[02d](02d_retention_and_reuse.ipynb) now puts the retention diagnosis first, explains every baseline, and keeps earlier reuse studies as an appendix. A local normalization backward rule is implemented and tested below; no learning gate has been added.

## Two different sources of loss

The smaller shared-feature learner has 15 → 30 units. Making it 45 → 90 removes the observed task loss: absent-task RMSE improves from **1.012 to 1.008** over a million updates. Weights still move, so this establishes task retention, not literal feature stability.

With **separate lower teacher features**, a 90 → 90 learner still loses some accuracy: **1.018 → 1.032** over a million updates. Initial learning uses the established 10k-example task blocks. Alternating every example left this case poorly learned before the absence, so it was not a clean retention test.

The controls distinguish these cases:

| Control | Small, shared features; 200k absent examples | Wide, separate features; 500k absent examples |
| --- | ---: | ---: |
| Ordinary training | 1.113 | 1.027 |
| Remove target noise | 1.105 | 1.019 |
| Freeze lower layer | 1.103 | 1.018 |
| Freeze upper layer | 1.057 | 1.037 |
| Freeze both hidden layers | 1.039 | 1.018 |
| Saved-model targets + noise | 1.055 | 1.027 |
| Saved-model targets, no noise | 1.047 | 1.018 |

These are absent-task expected noisy RMSEs. The small case uses 30 paired seeds; the wide controls and width comparisons use five paired seeds. Both tasks are initially learned for 1.6M examples.

The absent readout is always fixed; the active readout learns. Freezing includes weights, bias and normalization state. Freezing the upper layer does not freeze its inputs. Autostep state still updates, but frozen weight updates are discarded.

In the small network, ongoing fitting of the active problem is implicated: substituting the saved model's active predictions reduces absent-task RMSE to 1.055 with noise. In the wide separate-feature network, **noise-driven lower-layer changes account for most of the remaining gap**. The saved-model target is already fitted at the start, yet adding noise reproduces the gap. This is the cleaner case for the next mechanism test.

## A concrete normalization issue

Our normalization correction accounts for changes caused by a unit's own weights, but misses changes caused by earlier layers. A standardized scalar has constant mean squared activation. Nevertheless, the two-layer backward pass gives a nonzero lower-layer gradient for that quantity; differentiating all normalization moments removes it.

The local custom backward now corrects incoming error before propagating it upstream, using per-unit error and error–activation traces. Switching to it at the same learned checkpoint reduced absent-task RMSE after 1M examples from **1.03241 to 1.02925**, against the preserved reference **1.01752** (five paired seeds). Active-task RMSE improved from **1.00881 to 1.00667**. This closes about **21% of the excess RMSE**, but does not solve forgetting. The exact-moment backward check passed; online traces remain an approximation. The comparison took 61 seconds on GPU.

## Why Autostep does not simply freeze

Autostep uses current gradient history, not past usefulness. Consistent gradients can increase step sizes; weakly constrained directions need not provide enough opposing gradients to shrink them quickly. In the wide separate-feature control, median step sizes fall only about 4% and 5% across the two hidden layers in 500k updates. That still permits substantial movement. More directly, the units matching the absent task's ten lower features have essentially unchanged median incoming-weight rates after 1M updates: ratios 0.9987–1.0017. Their mean firing correlation with those true features falls from 0.998 to 0.993.

The deep-network implementation also uses squared source inputs as a curvature approximation. Its contribution has not been isolated. A persistent gradient would not, by itself, distinguish a genuinely obsolete feature from interference by another task.

**Next:** explain the remaining noise-driven drift; the local backward correction alone does not resolve it. Training with the new rule from initialization and indefinite retention remain untested.

## Fixed true lower-feature control

With the lower representation exact and fixed, absent-task RMSE rises from **1.080** to **1.255** at 1M and **1.309** at 4M examples. Active-task RMSE improves from **1.078** to **1.008**. The preserved reference stays at **1.080**.

This isolates a failure in the learned upper representation; changing lower features are not necessary for forgetting. It does not yet distinguish noise-driven drift from interference during further fitting, or establish an asymptotic trend. The upper layer has 30 units against a sufficient construction using 20, but the learner need not discover that construction. Next isolate noise versus continued fitting before deciding whether protection is warranted.

## Small-network cause identified

**Explanation.** The learned upper representation is still an approximation. Further fitting changes features the absent task also needs. There is room for an exact representation, but the learner has not found it.

| Control | Absent-task RMSE after 500k |
| --- | ---: |
| Preserved representation | 1.094 |
| Ordinary learning | 1.205 |
| No target noise | 1.195 |
| Already-fitted target + noise | 1.096 |
| Already-fitted target, no noise | 1.094 |
| 01 backward rule | 1.205 |

The active readout is not stuck: with upper features frozen it reaches 1.067, against a fitted-readout reference of 1.067. With ordinary learning the active task improves to 1.010, but refitting the absent readout only brings its error down to 1.149, versus 1.090 before the absence. Some loss is therefore in the features themselves, not just a mismatch with the old readout.

Given exact upper features and readouts, the same small network starts at RMSE 1 and ends at 1.002 for the active task and 1.001 for the absent task. Capacity is sufficient for stable performance over this test; the difficulty is how learning organizes and changes that capacity.

This is consistent with the mechanism behind 01: standardization suppresses noise-driven feature death; it does not prevent useful features from changing to reduce systematic error on another task. The same failure with the 01 rule also rules out the new backward approximation as its main cause here. Why learning does not discover the exact representation, and indefinite retention, remain separate questions. No gate is justified by this test alone.
