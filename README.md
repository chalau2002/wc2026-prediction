# WC 2026 Monte Carlo Simulator

This project builds an end-to-end prediction pipeline for the 2026 FIFA World Cup. It combines historical international match data, Elo ratings, engineered football features, machine learning goal models, and a Monte Carlo tournament simulator to estimate each team's probability of reaching every stage of the competition.

The workflow starts by calculating historical Elo ratings from past national-team results. These ratings are used as a compact measure of team strength and are combined with additional features such as recent scoring form, goals conceded, head-to-head context, World Cup experience, competition type, and dynamic tournament information. Together, these variables form the training dataset used by the prediction model.

The modelling step trains two XGBoost models with a Poisson objective: one model predicts the expected goals for the home team, and the other predicts the expected goals for the away team. These expected-goal values are treated as Poisson lambdas, which makes them suitable for generating realistic football scorelines instead of only predicting win/draw/loss outcomes.

The simulation step then runs the full 2026 World Cup many times using Monte Carlo. For each simulated match, the model predicts expected goals, actual goals are sampled from Poisson distributions, and the tournament state is updated. This includes group standings, knockout qualification, Elo changes, rolling form, head-to-head history, and other dynamic features. Repeating this process thousands of times produces probabilities for each team to win the tournament, reach the final, qualify from the group stage, and progress through every knockout round.

## Pipeline

```text
src/elo.py                       -> calculates historical Elo and final_elo.csv
src/features.py                  -> creates data/processed/training_df.csv
src/model.py                     -> trains XGBoost Poisson models for home_score and away_score
src/simulate.py                  -> simulates the 2026 World Cup with dynamic features
src/run_all.py                   -> runs the full pipeline in order
scripts/build_dashboard_data.py  -> builds the web interface data
```

## Data

Main inputs:

```text
data/raw/results.csv
data/raw/groups.csv
data/raw/knockout_slots.csv
data/raw/fw26_best_third_placed_combinations.csv
```

The simulator assumes that the last 72 rows in `data/processed/training_df.csv` are the 2026 World Cup group-stage matches, still without results.

## Running

Full pipeline:

```bash
python3 src/run_all.py --n-sims 1000
```

Simulation only, using existing models and data:

```bash
python3 src/simulate.py --n-sims 1000 --output-dir data/processed/monte_carlo_runs/manual_1000
```

Individual commands:

```bash
python3 src/elo.py
python3 src/features.py
python3 src/model.py
python3 src/simulate.py --n-sims 1000
```

## Outputs

`run_all.py` creates one folder per run:

```text
data/processed/monte_carlo_runs/YYYYMMDD_HHMMSS/
```

Main generated files:

```text
wc2026_monte_carlo_summary_<timestamp>.csv
wc2026_group_sample_<timestamp>.csv
wc2026_knockout_sample_<timestamp>.csv
wc2026_dynamic_feature_sample_<timestamp>.csv
logs/
  1_elo.log
  2_features.log
  3_model.log
  4_simulate.log
```

The Monte Carlo summary includes probabilities such as:

```text
p_round_of_32
p_round_of_16
p_quarter_final
p_semi_final
p_final
p_champion
p_runner_up
p_third_place
```

`wc2026_group_sample_*.csv` stores the simulated group-stage matches.

`wc2026_dynamic_feature_sample_*.csv` stores a sample of simulated matches with the features used before each prediction, including group-stage and knockout matches.

## Interface

The web interface is in `public/` and shows the champion ranking, team probabilities, groups, and most likely bracket.

After running a simulation, generate the JSON used by the interface. By default, the script uses the latest Monte Carlo run folder in `data/processed/monte_carlo_runs/`:

```bash
python3 scripts/build_dashboard_data.py
```

You can still point the interface at a specific run:

```bash
python3 scripts/build_dashboard_data.py \
  --summary data/processed/monte_carlo_runs/<run>/wc2026_monte_carlo_summary_<timestamp>.csv \
  --group data/processed/monte_carlo_runs/<run>/wc2026_group_sample_<timestamp>.csv \
  --knockout data/processed/monte_carlo_runs/<run>/wc2026_knockout_sample_<timestamp>.csv
```

The page also refreshes `dashboard-data.json` automatically at runtime, so it can pick up a newly regenerated build without a manual reload.

Serve the interface locally:

```bash
python3 -m http.server 8000 --directory public
```

Then open `http://localhost:8000`.

## Interpretability

`src/model.py` generates interpretability artifacts in `artifacts/<timestamp>/`:

```text
feature_importance_home.png
feature_importance_away.png
shap_summary_home.png
shap_summary_away.png
audit_predictions_<timestamp>.csv
```

`feature_importance_home.png` and `feature_importance_away.png` show the global feature importance reported by XGBoost for each model.

`shap_summary_home.png` and `shap_summary_away.png` show feature impact on predictions with SHAP. These plots help explain not only which features matter, but also whether high or low feature values push goal predictions up or down.

`audit_predictions_<timestamp>.csv` stores model predictions on the full training dataset, including lambdas and estimated 1X2 probabilities.

## Dynamic Features

During the simulation, features are recalculated before each match. After each simulated result, the tournament state is updated:

- Elo ratings for both teams;
- rolling averages for goals scored and conceded;
- head-to-head history;
- World Cup experience;
- group standings, when applicable.

This means the same match can have different lambdas across simulations, depending on previous results in that simulation.

## Model

Training uses two XGBoost models with a Poisson objective:

- `model_home`: estimates `home_score`;
- `model_away`: estimates `away_score`.

Predictions are lambdas, meaning expected goals. In the simulation, actual goals are sampled as:

```text
home_goals ~ Poisson(lambda_home)
away_goals ~ Poisson(lambda_away)
```

## Runtime

Each simulation has 104 matches:

```text
72 group-stage matches + 32 knockout matches
```

Therefore:

```text
1,000 simulations  = 104,000 simulated matches
10,000 simulations = 1,040,000 simulated matches
```

Because features are dynamic, very high `--n-sims` values can take a long time. Use `100` or `1000` to test changes; use `10000` or more for more stable results.

## Reproducibility

The scripts accept a seed:

```bash
python3 src/run_all.py --n-sims 1000 --seed 42
python3 src/simulate.py --n-sims 1000 --seed 42
```

With the same data, models, and seed, results should be reproducible.
