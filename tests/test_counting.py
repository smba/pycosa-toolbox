import itertools
import random

import numpy as np
import pytest
from scipy.stats import chisquare

from pycosa import sampling
from pycosa.counting import ModelCounter
from pycosa.modeling import VariabilityModel
from conftest import all_valid


def brute_force_count(clauses, n):
    X = np.array(list(itertools.product([False, True], repeat=n)), dtype=bool)
    ok = np.ones(len(X), dtype=bool)
    for clause in clauses:
        sat = np.zeros(len(X), dtype=bool)
        for lit in clause:
            col = X[:, abs(lit) - 1]
            sat |= col if lit > 0 else ~col
        ok &= sat
    return int(ok.sum())


def test_count_random_cnfs():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 9)
        clauses = [
            [rng.choice([-1, 1]) * rng.randint(1, n) for _ in range(rng.randint(1, 3))]
            for _ in range(rng.randint(0, 14))
        ]
        assert ModelCounter(clauses, range(1, n + 1)).count() == brute_force_count(
            clauses, n
        )


def test_count_h2_matches_brute_force(h2):
    n = len(h2.get_binary_features())
    assert h2.count_configurations() == brute_force_count(h2.dimacs, n) == 40960


def test_count_with_assumptions_and_marginals():
    # a -> b, exactly: {}, {b}, {a, b}
    mc = ModelCounter([[-1, 2]], [1, 2])
    assert mc.count() == 3
    assert mc.count([1]) == 1
    assert mc.count([-2]) == 1
    assert mc.marginals() == pytest.approx({1: 1 / 3, 2: 2 / 3})


def test_unsatisfiable():
    mc = ModelCounter([[1], [-1]], [1])
    assert mc.count() == 0
    with pytest.raises(ValueError):
        mc.sample()


def test_option_frequencies_invalidated_by_select():
    vm = VariabilityModel()
    vm.from_dimacs([[-1, 2]], {1: "a", 2: "b"})
    assert vm.count_configurations() == 3
    vm.select(["a"])
    assert vm.count_configurations() == 1
    assert vm.option_frequencies() == {"a": 1.0, "b": 1.0}


def _toy_model():
    # 4 options: a xor b, c -> a, d free  =>  (a, c in {0,1}) + b  => 3 * 2 = 6
    vm = VariabilityModel()
    vm.from_dimacs([[1, 2], [-1, -2], [-3, 1]], {1: "a", 2: "b", 3: "c", 4: "d"})
    return vm


def test_uniform_sampler_is_uniform():
    vm = _toy_model()
    assert vm.count_configurations() == 6
    df = sampling.UniformSampler(vm, seed=42).sample(6000, unique=False)
    assert all_valid(vm, df.drop_duplicates())
    counts = df.value_counts()
    assert len(counts) == 6
    # goodness of fit against the uniform distribution
    assert chisquare(counts.to_numpy()).pvalue > 0.001


def test_uniform_sampler_unique_and_exhaustive():
    vm = _toy_model()
    df = sampling.UniformSampler(vm, seed=0).sample(10)
    assert len(df) == 6
    assert not df.duplicated().any()


def test_uniform_sampler_real_models(vm):
    df = sampling.UniformSampler(vm, seed=1).sample(40)
    assert len(df) == 40
    assert not df.duplicated().any()
    assert all_valid(vm, df)
    a = sampling.UniformSampler(vm, seed=3).sample(5)
    b = sampling.UniformSampler(vm, seed=3).sample(5)
    assert a.equals(b)


def test_uniform_sample_matches_option_frequencies(vm):
    df = sampling.UniformSampler(vm, seed=5).sample(2000, unique=False)
    freq = vm.option_frequencies()
    observed = df.mean()
    for f in vm.get_binary_features():
        assert abs(observed[f] - freq[f]) < 0.05, f
