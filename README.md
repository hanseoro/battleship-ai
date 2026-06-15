# Battleship AI

An exploration of different algorithms for solving a randomized standard Battleship board. 

## Agents included

- `random`: baseline agent with random exploration and local follow-up after a hit
- `checkerboard`: parity-based search plus local follow-up
- `heatmap`: probability-style placement counting
- `bayesian_mc`: Bayesian search with Monte Carlo sampling

## Dependencies

The project is light on dependencies:

- Python `3.11+`
- `pytest` for running the tests
- `setuptools` and `wheel` for editable install support

There are no extra runtime libraries listed in `pyproject.toml` right now.

## How to check dependencies

Make sure Python is new enough:

```bash
python3 --version
```

You should see Python `3.11` or higher.

To check whether `pytest` is installed:

```bash
python3 -m pytest --version
```

If you want to inspect installed packages more generally:

```bash
python3 -m pip list
```

## How to install dependencies

If you only want the project itself:

```bash
python3 -m pip install -e .
```

If you also want the testing tools:

```bash
python3 -m pip install -e '.[dev]'
```

If your environment is old and pip complains, upgrading these usually helps:

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

## How to run the project

Run the default simulation:

```bash
python3 -m battleship
```

Run a specific agent:

```bash
python3 -m battleship --agent checkerboard --games 50 --seed 42
```

Available agent names:

- `random`
- `checkerboard`
- `heatmap`
- `bayesian_mc`

For per-game output:

```bash
python3 -m battleship --agent heatmap --games 5 --seed 42 --per-game
```

For a full shot-by-shot trace:

```bash
python3 -m battleship --agent random --games 1 --seed 42 --verbose
```

## How to run tests

Run the whole test suite:

```bash
python3 -m pytest -q
```

At the moment, the tests cover the engine, the baseline agent, the advanced agents, and repeated simulation behavior.

## How to run larger evaluations

For report-style evaluation across multiple agents:

```bash
python3 testing/evaluate_agents.py --agents random checkerboard heatmap --games 1000 --seed 42 --fresh
```

This script writes:

- per-game CSV results
- summary CSV results
- summary JSON results

Example including all four agents:

```bash
python3 testing/evaluate_agents.py --agents random checkerboard heatmap bayesian_mc --games 100 --seed 42 --fresh
```

One important warning: the Bayesian Monte Carlo agent is much slower than the other three, so it is smart to test it with smaller counts first before launching a huge run.

## Relevant Literature

https://is.muni.cz/th/oupp1/Reinforcement_Learning_for_the_Game_of_Battleship.pdf  
https://link.springer.com/chapter/10.1007/978-3-319-00542-3_20  
https://arxiv.org/abs/2004.07354  
