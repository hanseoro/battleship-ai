# Battleship AI Project

This project is our Battleship simulator and attack-agent tester for CS 175. The main idea is simple: we hide a standard Battleship board, let different AI agents guess shots one turn at a time, and then compare how efficiently they finish the game. We used this setup to see how much smarter strategies actually help compared to a basic random-style baseline.

What makes the project interesting is that every agent sees only partial information. An agent only knows which shots already happened, which ones missed, which ones hit, and which ships have been sunk. From there, it has to decide the next best shot without seeing the full board.

## How the project works

The code is split into a few small parts:

- `src/battleship/engine.py` runs a single Battleship game, checks ship placement legality, and handles hit, miss, and sunk logic.
- `src/battleship/simulation.py` runs many games in a row and computes summary stats.
- `src/battleship/agent_random.py` is the baseline agent. It explores randomly, but after landing a hit it follows nearby cells to try to finish the ship.
- `src/battleship/agent_checkerboard.py` uses parity, so it mainly shoots checkerboard-pattern cells first.
- `src/battleship/agent_heatmap.py` scores cells based on how many legal ship placements pass through them.
- `src/battleship/agent_bayesianMC.py` samples possible remaining boards and fires at the cell that appears most often.
- `testing/evaluate_agents.py` runs larger experiments and saves per-game and summary results.

In short, the flow is:

1. Randomly place ships on the board.
2. Give the chosen agent the current observation.
3. Let the agent pick a shot.
4. Update the board with hit, miss, or sunk.
5. Repeat until all ships are sunk.
6. Save the total attacks and compare agents over many games.

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

If you want per-game output:

```bash
python3 -m battleship --agent heatmap --games 5 --seed 42 --per-game
```

If you want a full shot-by-shot trace:

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

## Current status

Right now the codebase is in a good place for running and comparing the four attack agents. The fast agents are easy to scale into the thousands of games. The main practical limit is the Bayesian Monte Carlo agent, which is much more expensive per game than the others.
