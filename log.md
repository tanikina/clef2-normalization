# Experimentation Log

This file is meant to log the development progress of the shared task. Please log relevant experiments including a short motivation, the commands, and a summary of the obtained results or follow-up ideas.

**Usage:** If you want to add content, please create a new section with the current date as heading (if not yet available), and structure your content e.g. as follows:

```markdown
## 2025-03-27

### CLEF Baseline: Learning Rate Optimization

- training the models on the English data and testing learning rates in the following range: 1e-3, 5e-4, 3e-4, 1e-4, for each learning rate we train 3 models with different seeds.
  - command: see `scripts/baselines_cluster/clef-baseline-lr-search.sh`
  - wandb (weights & biases) project: https://wandb.ai/tanikina/clef2-normalization-src_baselines (this is just an example!)
  - metric values:
    |         | val-official-METEOR | val-sampled-METEOR | combined-val-METEOR |
    |---------|---------------------|--------------------|---------------------|
    | lr-1e-3 | 0.555               | 0.600              | 0.578               |
    | lr-5e-4 | .....               | .....              | .....               |
    | lr-3e-4 | .....               | .....              | .....               |
    | lr-1e-4 | .....               | .....              | .....               |
  - outcome: the best learning rate is ...
```
