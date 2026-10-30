# Emergency-domain evaluation set

**This directory contains no sentences.** The three `*_TEMPLATE.tsv` files are header-only.
The emergency-domain test set used for the revised manuscript is not part of this repository.

Formats expected by the scripts (tab-separated, UTF-8, one header row):

| file | columns | used by |
|---|---|---|
| `emergency_set_*.tsv` | `id`, `de`, `en` (cased reference) | `scripts/evaluate.py --tsv` |
| `classifier_eval_*.tsv` | `id`, `de`, `label` (Medical/Fire/Disaster/Police/Neutral, human-verified) | `scripts/eval_classifier.py` |
| `error_annotations_*.tsv` | `id`, `system`, then 0/1 flags: entity, compound, word_order, omission, hallucination | `scripts/error_analysis.py` |

Document here, once the set is released: how sentences were selected/written, who translated and verified them,
annotator count and agreement, licence, and whether any sentence overlaps with the training data.
