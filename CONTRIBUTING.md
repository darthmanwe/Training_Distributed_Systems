# Contributing

## Adding a new environment

1. Create `src/workers/envs/your_env.py` implementing the Gymnasium interface
2. Add an `elif` branch in `src/workers/rollout_worker.py::_make_env()`
3. Add a config preset in `configs/your_env.yaml`
4. Test: `python scripts/run_local.py --config configs/your_env.yaml`

## Adding a new RL algorithm

1. Create `src/trainer/your_algo.py` implementing `TrainerInterface` from `src/trainer/interface.py`
2. Required methods: `update()`, `get_policy_state_dict()`, `load_policy_state_dict()`, `get_checkpoint_state()`, `load_checkpoint_state()`
3. Add selection logic in `scripts/run_local.py` based on `config.trainer.algorithm`
4. Add unit tests in `tests/test_your_algo.py`

## Adding a new metric

1. Add the instrument in `src/infra/metrics.py::setup_metrics()`
2. Call `record("your_metric_name", value)` where appropriate
3. Add a panel in `docker/grafana/dashboards/training.json`
4. Document in the README metrics table

## Code standards

- Type annotations on all public functions
- Run `ruff check` and `mypy src/` before committing
- Unit tests for algorithm correctness, integration tests for pipeline
- Config changes go through Pydantic validation
