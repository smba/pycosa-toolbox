import itertools

import numpy as np
import pandas as pd
import pytest
import z3

from pycosa import sampling
from conftest import all_valid


def test_dfs_valid_unique(vm):
    df = sampling.DFSSampler(vm, seed=1).sample(25)
    assert df.shape == (25, len(vm.get_binary_features()))
    assert list(df.columns) == vm.get_binary_features()
    assert not df.duplicated().any()
    assert all_valid(vm, df)


def test_dfs_reproducible(vm):
    a = sampling.DFSSampler(vm, seed=7).sample(10)
    b = sampling.DFSSampler(vm, seed=7).sample(10)
    pd.testing.assert_frame_equal(a, b)


def test_dfs_exhausts_small_space():
    from pycosa.modeling import VariabilityModel

    vm = VariabilityModel()
    vm.from_dimacs([[1], [2, 3]], {1: "a", 2: "b", 3: "c"})
    df = sampling.DFSSampler(vm, seed=0).sample(100)
    assert len(df) == 3


def test_distance_based_spreads_distances(vm):
    df = sampling.DistanceBasedSampler(vm, seed=1).sample(30)
    assert len(df) == 30
    assert not df.duplicated().any()
    assert all_valid(vm, df)
    # actually distance-based: many different numbers of selected options
    assert df.sum(axis=1).nunique() >= 5


@pytest.mark.parametrize("t", [1, 2])
@pytest.mark.parametrize("negative_wise", [False, True])
@pytest.mark.parametrize("extend", [False, True])
def test_coverage_complete(vm, t, negative_wise, extend):
    df = sampling.CoverageBasedSampler(vm, seed=1).sample(
        t=t, negative_wise=negative_wise, extend=extend
    )
    assert all_valid(vm, df)
    assert not df.duplicated().any()

    X = df.to_numpy(dtype=bool)
    if negative_wise:
        X = ~X
    s = z3.Solver()
    s.add(vm.get_binary_clauses())
    V = vm.get_vars()
    for combo in itertools.combinations(range(X.shape[1]), t):
        if X[:, list(combo)].all(axis=1).any():
            continue
        lits = [z3.Not(V[i]) if negative_wise else V[i] for i in combo]
        assert s.check(*lits) == z3.unsat, combo


def test_coverage_greedy_is_small(vm):
    df = sampling.CoverageBasedSampler(vm, seed=1).sample(t=2)
    assert len(df) <= 15


def test_group_sampler_pairs(vm):
    features = vm.get_binary_features()
    core, dead = vm.core_and_dead_features()
    optional = [f for f in features if f not in core and f not in dead]
    group = optional[:2]
    en, dis = sampling.GroupSampler(vm, seed=1).sample(group, 10)
    assert len(en) == len(dis) > 0
    assert all_valid(vm, en) and all_valid(vm, dis)
    assert en[group].all().all() and not dis[group].any().any()
    others = [f for f in features if f not in group]
    pd.testing.assert_frame_equal(en[others], dis[others])
    assert not en.duplicated().any()


def test_group_sampler_unknown_option(h2):
    with pytest.raises(ValueError):
        sampling.GroupSampler(h2).sample(["nope"], 3)


def test_offline_sampler():
    df = pd.DataFrame({"a": [1, 0, 1, 0], "b": [0, 0, 1, 1], "perf": [1, 1, 2, 2]})
    en, dis = sampling.OfflineSampler(df.drop(columns="perf")).elementary_effect_sample(
        ["a"]
    )
    assert sorted(zip(en, dis)) == [(0, 1), (2, 3)]
