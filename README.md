# sr-tire

Utilities for:

- loading and processing tire test data
- plotting measured tire data together with the force model
- fitting a symbolic `mu(v)` expression with Flex symbolic regression

## Layout

- `src/sr_tire/process_data.py`: load and split the tire dataset
- `src/sr_tire/force.py`: force-model implementation
- `src/sr_tire/plot.py`: plotting utilities
- `src/sr_tire/models/flex_sr_run.py`: Flex symbolic regression entry point
- `src/sr_tire/models/config.yaml`: Flex configuration
- `data/lateral_tire_test.csv`: tire dataset

## Install

From the repo root:

```bash
python -m pip install -e .
```

## Make Plots

To plot the default force model against the processed tire data:

```bash
python -m sr_tire.plot
```

## Run Flex SR

To run a single training run without hyperparameter optimization:

```bash
python -m sr_tire.models.flex_sr_run
```

To run multiple independent training runs:

```bash
python -m sr_tire.models.flex_sr_run --num-runs 5
```

To control the base random seed used for repeated runs:

```bash
python -m sr_tire.models.flex_sr_run --num-runs 5 --seed 123
```

To run with Optuna HPO:

```bash
python -m sr_tire.models.flex_sr_run --hpo
```

## Outputs

After running Flex SR, the script:

- prints dataset summary information
- fits the symbolic `mu(v)` model
- saves the best overall model metrics and expression to:

```text
src/sr_tire/models/best_model_results.txt
```

- saves the best overall plot to:

```text
src/sr_tire/models/best_model_plot.png
```

- saves the best overall `mu(v)` plot to:

```text
src/sr_tire/models/best_mu_plot.png
```

- when `--num-runs` is greater than `1`, also saves per-run artifacts under:

```text
src/sr_tire/models/flex_runs/
```

- writes one subdirectory per run, for example:

```text
src/sr_tire/models/flex_runs/run_001/best_model_results.txt
src/sr_tire/models/flex_runs/run_001/best_model_plot.png
src/sr_tire/models/flex_runs/run_001/best_mu_plot.png
src/sr_tire/models/flex_runs/run_001/train_mse_history.csv
src/sr_tire/models/flex_runs/run_001/val_mse_history.csv
```

- writes an aggregate run summary to:

```text
src/sr_tire/models/flex_runs/summary.txt
```

- writes a LaTeX summary document with median metrics and the best-test-`R^2` run to:

```text
src/sr_tire/models/flex_runs/summary.tex
```

## Notes

- The current dataset split is:
  - bins 1-3: training
  - bin 4: validation
  - bin 5: test
- The CSV is loaded from `data/` at the repo root.
