# pycosa-toolbox

[![Run tests](https://github.com/smba/pycosa-toolbox/actions/workflows/tests.yml/badge.svg?event=push)](https://github.com/smba/pycosa-toolbox/actions/workflows/tests.yml)
[![Docs](https://github.com/smba/pycosa-toolbox/actions/workflows/docs.yml/badge.svg)](https://github.com/smba/pycosa-toolbox/actions/workflows/docs.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

**pycosa** is a toolbox for sampling and analyzing configurations of
variability models. It models a system's configuration space as a
propositional formula over binary configuration options and provides
several sampling strategies (including exact uniform sampling), exact
configuration counting, and metrics to assess and compare sample quality.

Full API documentation is generated from the docstrings and published at
**[GitHub Pages](https://smba.github.io/pycosa-toolbox/)**.

## Installation

Install the latest version from GitHub:

```bash
pip install git+https://github.com/smba/pycosa-toolbox.git@main
```

Update an existing installation:

```bash
pip install --upgrade git+https://github.com/smba/pycosa-toolbox.git@main
```

Requires Python 3.9 or newer. For development (editable install with test
and documentation dependencies):

```bash
git clone https://github.com/smba/pycosa-toolbox.git
cd pycosa-toolbox
pip install -e ".[test,doc]"
```

## Overview

| Module | Purpose |
| --- | --- |
| `pycosa.modeling` | Propositional variability models (CNF) and their analysis |
| `pycosa.parsing` | Parsers for DIMACS and SPL Conqueror (XML) models |
| `pycosa.sampling` | Sampling strategies for configurations |
| `pycosa.counting` | Exact model counting and uniform sampling |
| `pycosa.metrics` | Sample-quality metrics (validity, diversity, coverage, ...) |
| `pycosa.learning` | Identifying influential and interacting options |
| `pycosa.util` | Preparing measurement data (collinearity, one-hot, interactions) |
| `pycosa.plotting` | Plotting distributions (e.g., mirrored histograms) |
| `pycosa.experiments` | Synthetic ground-truth performance models |
| `pycosa.compatibility` | Readers for third-party data (SPL Conqueror measurements) |

## Usage

### Loading a variability model

```python
from pycosa.modeling import VariabilityModel

vm = VariabilityModel.from_dimacs_file("_test_data/feature_models/sqlite.dimacs")

vm.is_satisfiable()          # False if the model has no valid configuration
vm.count_configurations()    # exact number of valid configurations
core, dead = vm.core_and_dead_features()
frequencies = vm.option_frequencies()  # fraction of configs selecting each option
```

SPL Conqueror models (XML) can be read with `pycosa.parsing.SPLCParser`:

```python
from pycosa.parsing import SPLCParser
from pycosa.modeling import VariabilityModel

parser = SPLCParser()
parser.parse("_test_data/splc/berkeleydb.xml")
vm = VariabilityModel()
vm.from_dimacs(parser.get_clauses(), parser.get_index2features())
```

### Sampling configurations

All samplers take the model plus an optional `seed` for reproducibility,
and return a `pandas.DataFrame` with one boolean column per option and one
row per configuration.

```python
from pycosa.modeling import VariabilityModel
from pycosa import sampling

vm = VariabilityModel.from_dimacs_file("_test_data/feature_models/sqlite.dimacs")

# distinct valid configurations, spread out via randomized solver phases
df = sampling.DFSSampler(vm, seed=1).sample(50)

# distance-based sampling: uniform over the number of selected options
df = sampling.DistanceBasedSampler(vm, seed=1).sample(50)

# uniform random sampling: every valid configuration is equally likely
df = sampling.UniformSampler(vm, seed=1).sample(50)

# greedy t-wise coverage (e.g., pair-wise); with polarity="both", every valid
# value combination of every pair of options appears in the sample
df = sampling.CoverageBasedSampler(vm, seed=1).sample(t=2, polarity="both")

# pairs of configurations differing exactly in a group of options
enabled, disabled = sampling.GroupSampler(vm, seed=1).sample(["OPTION_A", "OPTION_B"], size=20)
```

### Assessing and comparing samples

`pycosa.metrics` quantifies how good a sample is; `compare` turns the
metrics of several samples into a data frame:

```python
from pycosa import metrics

report = metrics.compare(vm, {
    "dfs": sampling.DFSSampler(vm, seed=1).sample(50),
    "distance": sampling.DistanceBasedSampler(vm, seed=1).sample(50),
    "uniform": sampling.UniformSampler(vm, seed=1).sample(50),
})
print(report)  # validity, uniqueness, coverage, diversity, frequency error
```

### Finding influential and interacting options

Combine `GroupSampler` with measured performances and `GroupLearner` to
screen options via group-wise elementary effects:

```python
from pycosa.learning import GroupLearner

learner = GroupLearner(vm.get_binary_features(), t1=0.5, t2=0.1, seed=1)
group = learner.suggest_options(size=5)
# measure the pair `enabled, disabled` returned by GroupSampler for `group`,
# then record the measurements:
# learner.classify(group, perf_enabled, perf_disabled)
statistics = learner.get_statistics()
```

Synthetic ground-truth performance models for controlled experiments are
available in `pycosa.experiments`:

```python
from pycosa.experiments import AttributedVariabilityModelGenerator

system = AttributedVariabilityModelGenerator(vm, seed=1)
perf = system.get_performances(df.to_numpy(dtype=bool))
```

### Command line

There is no command-line interface; pycosa is a library.

## Development

```bash
pip install -e ".[test]"   # editable install with test dependencies
python -m pytest          # run the test suite
pip install -e ".[doc]"   # with documentation dependencies
sphinx-build -M html docs docs/_build   # build the HTML documentation locally
black .                   # formatting is enforced by CI
```

## Documentation

The API documentation is built with Sphinx from the docstrings and
published to GitHub Pages on every push to `main`:
<https://smba.github.io/pycosa-toolbox/>

## License

This project is licensed under the GNU Affero General Public License v3;
see [LICENSE](LICENSE) for details.
