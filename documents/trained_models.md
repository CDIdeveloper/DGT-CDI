# Trained models

A registry of trained DGT models — what config produced them, where the checkpoint lives, and how they did on the held-out test set. Add one entry per model you want to keep / cite / deploy from.

For the routine that produces these models, see [modeling_routine.md](modeling_routine.md).

## Quick index

| Date | Config | Dataset | Seed | Best-val | Test | Checkpoint |
|---|---|---|---|---|---|---|
| _(none yet — add rows as you train)_ |

## Entry template

Copy the block below for each new model, fill in the fields, and append under "Models" further down.

```markdown
### <model_name>  (yyyy-mm-dd)

- **Config:** [<path/to/config.yaml>](../<path/to/config.yaml>)
- **Dataset:** <name>, split: <scaffold / standard / ...>
- **Train command:**
  ```bash
  python main.py --cfg <config> --repeat <N> seed 0 wandb.use False
  ```
- **Git SHA at train time:** `<short_sha>`  (`git rev-parse --short HEAD`)
- **Seed(s) reported:** <0 / median over 0..3 / best of 4>
- **Best epoch (by val):** <K>
- **Validation metric:** <metric_best>=<value>
- **Test metric:** <metric>=<value>  (full numbers in `<run_dir>/plots/summary.json`)
- **Checkpoint:** `<run_dir>/ckpt/<K>.ckpt`
- **Plots:** `<run_dir>/plots/`
- **Notes:** <class imbalance / manual stopping / unusual losses / anything weird>
```

## Models

_(no entries yet)_

---

## Conventions

- **Granularity.** One entry per *seed* that you intend to keep around. For multi-seed runs you usually keep the seed whose test metric is closest to the aggregated mean (median test seed), but document whichever rule you used.
- **Storage.** Checkpoints stay under `results/DGT/<config_name>/<seed>/ckpt/`. If you need to share or deploy one, copy it out to a stable location (e.g. `models/<model_name>.ckpt`) and update the entry's `Checkpoint:` field.
- **Reproducibility.** Always record the git SHA at train time — config files, loader code, and `dgt_train.py` all evolve, and the same YAML can produce different results after a code change.
- **Sunset old entries** when a better model on the same dataset supersedes them. Either delete the entry or move it to a `## Archive` section at the bottom of this file.
