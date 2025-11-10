# Repository Guidelines

## Project Structure & Module Organization
Core Python package code lives under `hummingbot/`, covering connectors, strategies, utilities, and client services. Controller-based workflows reside in `controllers/`, while helper entrypoints sit in `scripts/` and `bin/` (for example, `start` and `hummingbot_quickstart.py`). Configuration samples and user-specific secrets belong in `conf/` (e.g., `conf/connectors/` and `conf/strategies/`), compiled artifacts appear in `build/`, and runtime logs are captured in `logs/`. Tests mirror the package layout inside `test/`.

## Build, Test, and Development Commands
Run `make install` to bootstrap dependencies via `./install`. Use `make build` (invokes `./compile`) after touching Cython modules. `make test` executes `coverage run -m pytest` with resource-heavy suites skipped; follow with `make run_coverage` to inspect HTML coverage reports. Launch the client with `./start` or script automated sessions using `./bin/hummingbot_quickstart.py -p <password>`. The controller proof-of-concept runs through `make run-v2 scenario=v2_with_controllers.py`.

## Coding Style & Naming Conventions
Write Python 3 code with 4-space indentation and explicit types when helpful. Format using `black` (line length 120) and organize imports with `isort` per `pyproject.toml`. Keep modules and packages in `snake_case`, classes in `PascalCase`, asynchronous helpers with `_async` suffixes, and strategy config files named `conf/strategies/<strategy_name>.yml`. Exchange connector identifiers stay lowercase (for example, `binance_perpetual`).

## Testing Guidelines
Place unit and integration tests under `test/` following the source layout and name files `test_*.py`. Use targeted runs such as `pytest test/hummingbot/connector/derivative/test_binance_perpetual.py` while iterating, then ensure `make test` passes before submitting. Mock external exchanges unless you supply API keys, and update or add coverage-focused tests whenever modifying core connectors or strategies.

## Commit & Pull Request Guidelines
Compose commit subjects in the imperative mood (`fix`, `chore`, `(feat) add order states`) and keep diffs scoped. Provide descriptive bodies for behavior changes, reference issues with `#`, and mention dependent Gateway updates. Pull requests should include a succinct summary, test evidence (command output or screenshots), configuration notes, and any migration steps for operators. Request reviews from connector owners or strategy maintainers tied to your change.

## Security & Configuration Tips
Store API keys solely in local `conf/` files and never commit secrets. Validate rate-limit or permission updates against exchange documentation before shipping. Adjust logging via `conf/hummingbot_logs.yml` during debugging and scrub sensitive data from logs before sharing them in issues or support channels.
