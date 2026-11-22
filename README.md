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

# greedy t-wise coverage (e.g. pair-wise); every valid pair of selected
# options appears in at least one configuration
df = sampling.CoverageBasedSampler(vm, seed=1).sample(t=2)

# pairs of configurations differing exactly in a group of options
enabled, disabled = sampling.GroupSampler(vm, seed=1).sample(["OPTION_A", "OPTION_B"], size=20)
```

All samplers return a `pandas.DataFrame` with one boolean column per option.
SPL Conqueror models can be read with `pycosa.parsing.SPLCParser`.
