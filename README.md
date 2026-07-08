# sr-tire

Utilities for:

- loading and processing tire test data
- plotting measured tire data together with the force model
- fitting a symbolic `mu(v)` expression with Flex symbolic regression

## Layout

- `src/sr_tire/process_data.py`: load and split the tire dataset
- `src/sr_tire/force.py`: force-model implementation
- `src/sr_tire/dynamic_pde/simulate_pde.py`: JAX translation of the dynamic PDE simulation
- `src/sr_tire/dynamic_pde/compare_pacejka.py`: compare PDE forces against Pacejka data
- `src/sr_tire/plot.py`: plotting utilities
- `src/sr_tire/steady_stribeck/flex_sr_run.py`: Flex symbolic regression entry point
- `src/sr_tire/steady_stribeck/config.yaml`: Flex configuration
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

## Run Dynamic PDE Simulation

To run the dynamic PDE tire simulation:

```bash
python -m sr_tire.dynamic_pde.simulate_pde
```

The module saves the bristle deformation, force, and slip plots under the
repository-root `results/pde/`. To open
interactive plot windows as well:

```bash
python -m sr_tire.dynamic_pde.simulate_pde --show
```

To save those figures programmatically:

```python
from sr_tire.dynamic_pde.simulate_pde import plot_simulation, simulate_pde

result = simulate_pde()
plot_simulation(result, output_dir="results/pde", show=False)
```

To run the equivalent dctkit cochain-based spatial discretization with Diffrax
time integration:

```bash
python -m sr_tire.dynamic_pde.simulate_pde_dctkit
```

To compare the dynamic PDE forces with the comparable Pacejka force trajectory
in `data/Fx_dataset.csv`, using `parameterization=1`, `excitation=var2`,
`run=5`, `sigma_0=0.12`, amplitude ratio `0.4`, and `omega=10*pi`:

```bash
python -m sr_tire.dynamic_pde.compare_pacejka
```

This writes comparison plots and per-trajectory error metrics to the
repository-root `results/pde/`.

By default, the comparison plot shows that same `var2`, run 5 pair. To choose
another pair within `parameterization=1`:

```bash
python -m sr_tire.dynamic_pde.compare_pacejka --excitation var2 --run 1
```

To run Flex SR for an additive term in the dynamic PDE right-hand side:

```bash
python -m sr_tire.dynamic_pde.flex_sr_run
```

The run is configured by `src/sr_tire/dynamic_pde/config.yaml` and scores each
candidate by solving the PDE and comparing the resulting force trajectory with
the selected trajectory from `data/Fx_dataset.csv`.

## Run Flex SR

To run a single training run without hyperparameter optimization:

```bash
python -m sr_tire.steady_stribeck.flex_sr_run
```

To run multiple independent training runs:

```bash
python -m sr_tire.steady_stribeck.flex_sr_run --num-runs 5
```

To control the base random seed used for repeated runs:

```bash
python -m sr_tire.steady_stribeck.flex_sr_run --num-runs 5 --seed 123
```

To run with Optuna HPO:

```bash
python -m sr_tire.steady_stribeck.flex_sr_run --hpo
```

## Run Random Search Baseline

To run random expression search with the same primitive set as Flex SR and a
7-minute wall-clock budget per run:

```bash
python -m sr_tire.steady_stribeck.random_search_run
```

To run multiple independent random-search runs:

```bash
python -m sr_tire.steady_stribeck.random_search_run --num-runs 5
```

To change the per-run time budget:

```bash
python -m sr_tire.steady_stribeck.random_search_run --time-budget-seconds 420
```

## Results

After running Flex SR, the script:

- prints dataset summary information
- fits the symbolic `mu(v)` model
- saves all steady Stribeck artifacts under:

```
results/steady-stribeck/
```

Flex SR writes:

```text
results/steady-stribeck/best_model_results.txt
results/steady-stribeck/best_model_plot.png
results/steady-stribeck/best_mu_plot.png
results/steady-stribeck/flex_runs/
```

The random-search baseline writes analogous results to:

```text
results/steady-stribeck/random_search_best_model_results.txt
results/steady-stribeck/random_search_best_model_plot.png
results/steady-stribeck/random_search_best_mu_plot.png
results/steady-stribeck/random_search_runs/
```

## Notes

- The current dataset split is:
  - bins 1-3: training
  - bin 4: validation
  - bin 5: test
- The CSV is loaded from `data/` at the repo root.
