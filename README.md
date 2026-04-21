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
- `src/sr_tire/models/simple_sr.yaml`: Flex configuration
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

To run with Optuna HPO:

```bash
python -m sr_tire.models.flex_sr_run --hpo
```

## Outputs

After running Flex SR, the script:

- prints dataset summary information
- fits the symbolic `mu(v)` model
- saves model metrics and the best expression to:

```text
models/best_model_results.txt
```

- plots the best model together with the data at the end of the run

## Notes

- The current dataset split is:
  - bins 1-3: training
  - bin 4: validation
  - bin 5: test
- The CSV is loaded from `data/` at the repo root.
