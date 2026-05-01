# unlearn_wj2: Probe-Based Representation Erasure

This method avoids directly optimizing the evaluation truth-ratio metric.
It first trains lightweight probes that recover answer/paraphrase embeddings
from question-only hidden states, then updates selected model layers so the
probe can no longer recover the forget target while retain/utility behavior is
preserved.

Default training uses English forget supervision only. Target-language forget
sets remain evaluation-only to preserve the zero-shot cross-lingual setting.
