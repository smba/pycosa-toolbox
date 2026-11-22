# pycosa-toolbox [![Run tests](https://github.com/smba/pycosa-toolbox/actions/workflows/tests.yml/badge.svg?event=push)](https://github.com/smba/pycosa-toolbox/actions/workflows/tests.yml)

## Installation

Install the latest version:
```bash
pip install git+https://github.com/smba/pycosa-toolbox.git@main 
```

Update an existing installation:
```bash
pip install --upgrade git+https://github.com/smba/pycosa-toolbox.git@main 
```

For development (editable install with test dependencies):
```bash
pip install -e ".[test]"
python -m pytest
```

## Usage

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

# greedy t-wise coverage (e.g. pair-wise); with polarity="both", every valid
# value combination of every pair of options appears in the sample
df = sampling.CoverageBasedSampler(vm, seed=1).sample(t=2, polarity="both")

# pairs of configurations differing exactly in a group of options
enabled, disabled = sampling.GroupSampler(vm, seed=1).sample(["OPTION_A", "OPTION_B"], size=20)
```

All samplers return a `pandas.DataFrame` with one boolean column per option.
SPL Conqueror models can be read with `pycosa.parsing.SPLCParser`.

### Counting configurations

```python
vm.count_configurations()   # exact number of valid configurations
vm.option_frequencies()     # fraction of valid configurations selecting each option
```

### Comparing samples

```python
from pycosa import metrics

metrics.t_wise_coverage(vm, df, t=2).ratio  # share of valid pair-wise interactions covered

metrics.compare(vm, {
    "uniform": sampling.UniformSampler(vm, seed=1).sample(30),
    "distance": sampling.DistanceBasedSampler(vm, seed=1).sample(30),
    "pairwise": sampling.CoverageBasedSampler(vm, seed=1).sample(t=2, polarity="both"),
})
```

`compare` reports validity, uniqueness, 1- and 2-wise coverage, diversity
(mean and nearest-neighbour Hamming distance), and the deviation of option
frequencies from those of the whole configuration space.
