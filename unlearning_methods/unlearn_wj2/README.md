# unlearn_wj2: Probe-Based Representation Erasure

This method avoids directly optimizing the evaluation truth-ratio metric.
It first trains lightweight probes that recover answer/paraphrase embeddings
from question-only hidden states, then updates selected model layers so the
probe can no longer recover the forget target while retain/utility behavior is
preserved.

Default training uses English forget supervision only. Target-language forget
sets remain evaluation-only to preserve the zero-shot cross-lingual setting.

## Candidate construction

The method builds one global answer bank from the English forget split before
probe training. For each forget question, its answer and paraphrased answer are
marked as positives; answers from other forget samples and optional perturbed
answers are negatives. This avoids the degenerate batch-size-1 case where every
candidate in the batch is positive and the probe loss becomes exactly zero.

The default erasure objective minimizes the frozen probe's positive probability
for the current sample over this global bank (`erase.probe_loss=positive_prob`),
while retain/utility CE, KL, and hidden-preservation terms constrain collateral
damage.
