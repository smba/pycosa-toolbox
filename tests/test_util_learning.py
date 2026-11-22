import numpy as np
import pandas as pd
import pytest

from pycosa import learning, util
from pycosa.experiments import (
    AttributedVariabilityModelGenerator,
    SyntheticDataGenerator,
)
from pycosa.lib import OptionExhaustionError


def test_remove_multicollinearity():
    df = pd.DataFrame(
        {
            "mandatory": [1, 1, 1, 1],
            "A": [1, 0, 0, 1],
            "B": [0, 1, 0, 0],
            "C": [0, 0, 1, 0],
            "D": [1, 1, 0, 0],
            "perf": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = util.remove_multicollinearity(df)
    # constant column and one member of alternative group {A, B, C} removed
    assert list(out.columns) == ["B", "C", "D", "perf"]
    # input untouched
    assert "mandatory" in df.columns


def test_categorical_roundtrip_with_custom_index():
    df = pd.DataFrame({"x": ["a", "b", "a"], "y": [1, 2, 3]}, index=[10, 20, 30])
    enc = util.construct_categorical_variable(df, "x")
    assert list(enc.columns) == ["y", "x_a", "x_b"]
    assert enc.loc[20, "x_b"] and not enc.loc[20, "x_a"]
    dec = util.reconstruct_categorical_variable(enc, {"x_a": "a", "x_b": "b"}, "x")
    assert dec["x"].tolist() == ["a", "b", "a"]


def test_interaction_terms_custom_index():
    df = pd.DataFrame(
        {"cat": ["u", "v", "u"], "f": [1.0, 2.0, np.nan]}, index=[5, 6, 7]
    )
    out = util.construct_interaction_terms(df, "cat")
    assert out["u__f"].tolist() == [1.0, 0.0, 0.0]
    assert out["v__f"].tolist() == [0.0, 2.0, 0.0]


def test_group_learner():
    gl = learning.GroupLearner(["a", "b", "c"], seed=0)
    gl.classify(["a"], np.array([10.0, 10.0]), np.array([1.0, 1.0]))
    gl.classify(["b"], np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    stats = gl.get_statistics()
    assert stats.loc["a", "influential"] + stats.loc["a", "interacting"] == 1
    assert "b" not in gl.options
    assert set(gl.suggest_options(5)) == {"a", "c"}
    gl.classify(["a", "c"], np.array([1.0]), np.array([1.0]))
    with pytest.raises(OptionExhaustionError):
        gl.suggest_options()


def test_synthetic_generator_reproducible(h2):
    X = np.random.default_rng(0).integers(0, 2, size=(5, len(h2.get_binary_features())))
    a = AttributedVariabilityModelGenerator(h2, seed=4).get_performances(X)
    b = AttributedVariabilityModelGenerator(h2, seed=4).get_performances(X)
    np.testing.assert_allclose(a, b)
    gen = SyntheticDataGenerator(n_options=6, noise=0.0, seed=1)
    x = np.ones(6, dtype=bool)
    assert gen.get_performance(x) == pytest.approx(gen.get_performances(x[None, :])[0])


def test_mirrored_histogram(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from pycosa.plotting import mirrored_histogram

    rng = np.random.default_rng(0)
    out = tmp_path / "plot.png"
    mirrored_histogram(
        rng.normal(0, 1, 100),
        rng.normal(1, 1, 100),
        "a",
        "b",
        bandwith=0.3,
        export_name=str(out),
    )
    assert out.exists()
