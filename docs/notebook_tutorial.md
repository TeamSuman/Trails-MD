# Notebook tutorial

A runnable Jupyter notebook with **rendered plots** lives at
[`examples/notebooks/adaptive_msm_tutorial.ipynb`](https://github.com/TeamSuman/AutoSampler/blob/devel/examples/notebooks/adaptive_msm_tutorial.ipynb).
It uses small synthetic examples so every figure renders in seconds without
running molecular dynamics.

It covers, end to end:

1. **The input file** — load the annotated template and validate it against the
   schema.
2. **VAMP-2 feature selection** — rank candidate feature sets and pick the
   informative ones.
3. **MSM estimation** — build an MSM from a synthetic metastable chain and plot
   implied timescales and metastable free energies.
4. **Convergence detection** — watch the `ConvergenceMonitor` fire.
5. **Weighted-ensemble resampling** — split/merge with conserved weight.
6. **The analysis report** — synthesise a run directory and render the
   multi-panel `plot_convergence_report`.

## Run it yourself

```bash
pip install -e ".[deep-tica,examples]" jupyter
jupyter notebook examples/notebooks/adaptive_msm_tutorial.ipynb
```

Then continue with a real campaign using the [input file](input_file.md) and the
[adaptive-MSM tutorial](tutorials/adaptive_msm.md).
