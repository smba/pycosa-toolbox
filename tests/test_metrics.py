import pandas as pd
import pytest

from pycosa import metrics, sampling
from pycosa.modeling import VariabilityModel


@pytest.fixture
def ab():
    # a -> b; valid: {}, {b}, {a, b}
    vm = VariabilityModel()
    vm.from_dimacs([[-1, 2]], {1: "a", 2: "b"})
    return vm


def test_validity_and_uniqueness(ab):
    df = pd.DataFrame({"a": [True, True, False], "b": [True, False, False]})
    assert metrics.validity(ab, df) == pytest.approx(2 / 3)
    assert metrics.uniqueness(pd.concat([df, df.iloc[:1]])) == pytest.approx(3 / 4)


def test_t_wise_coverage_by_hand(ab):
    # valid pair-wise interactions: (a,b) in {00, 01, 11}; 10 is invalid
    df = pd.DataFrame({"a": [False], "b": [False]})
    r = metrics.t_wise_coverage(ab, df, t=2, return_missing=True)
    assert (r.covered, r.valid) == (1, 3)
    assert sorted(r.missing) == [
        (("a", False), ("b", True)),
        (("a", True), ("b", True)),
    ]
    # 1-wise: a=0, a=1, b=0, b=1 are all valid; sample covers a=0, b=0
    assert metrics.t_wise_coverage(ab, df, t=1).ratio == pytest.approx(0.5)
    assert metrics.t_wise_coverage(ab, df, t=1, polarity="positive").ratio == 0.0


def test_t_wise_coverage_ignores_column_order_and_extra_columns(ab):
    df = pd.DataFrame({"perf": [1.0], "b": [True], "a": [True]})
    assert metrics.t_wise_coverage(ab, df, t=2).covered == 1


def test_exhaustive_sample_has_full_coverage(ab):
    df = sampling.DFSSampler(ab, seed=0).sample(10)
    for t in (1, 2):
        assert metrics.t_wise_coverage(ab, df, t=t).ratio == 1.0


@pytest.mark.parametrize("t", [1, 2])
@pytest.mark.parametrize("polarity", ["positive", "negative", "both"])
def test_coverage_sampler_meets_metric(vm, t, polarity):
    df = sampling.CoverageBasedSampler(vm, seed=2).sample(t=t, polarity=polarity)
    assert metrics.t_wise_coverage(vm, df, t=t, polarity=polarity).ratio == 1.0


def test_diversity_and_distances():
    df = pd.DataFrame({"a": [0, 1, 1], "b": [0, 1, 0]}, dtype=bool)
    D = metrics.hamming_distances(df)
    assert D.tolist() == [[0, 2, 1], [2, 0, 1], [1, 1, 0]]
    d = metrics.diversity(df)
    # normalized distances: 2/2, 1/2, 1/2
    assert d["mean_distance"] == pytest.approx(2 / 3)
    assert d["mean_nearest"] == pytest.approx(0.5)
    assert metrics.distance_distribution(df).to_dict() == {0: 1, 1: 1, 2: 1}


def test_frequency_error(ab):
    # population frequencies: a = 1/3, b = 2/3
    exhaustive = pd.DataFrame({"a": [False, False, True], "b": [False, True, True]})
    assert metrics.option_frequency_error(ab, exhaustive) == pytest.approx(0.0)
    biased = pd.DataFrame({"a": [False], "b": [False]})
    assert metrics.option_frequency_error(ab, biased) == pytest.approx(
        (1 / 3 + 2 / 3) / 2
    )


def test_compare(vm):
    samples = {
        "uniform": sampling.UniformSampler(vm, seed=1).sample(20),
        "classic": sampling.DFSSampler(vm, seed=1, random_phase=False).sample(20),
    }
    table = metrics.compare(vm, samples)
    assert list(table.index) == ["uniform", "classic"]
    assert {"validity", "coverage_t1", "coverage_t2", "frequency_error"} <= set(
        table.columns
    )
    assert (table["validity"] == 1.0).all()
    # the clustered enumeration is clearly worse than uniform sampling
    assert table.loc["uniform", "coverage_t2"] > table.loc["classic", "coverage_t2"]
    assert (
        table.loc["uniform", "frequency_error"]
        < table.loc["classic", "frequency_error"]
    )
