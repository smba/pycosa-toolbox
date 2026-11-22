"""
Metrics for assessing and comparing configuration samples.

All functions take a sample as a ``pandas.DataFrame`` with one boolean column
per option (as returned by the samplers in ``pycosa.sampling``). Functions
that need the variability model take it as first argument.
"""

import itertools
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import z3

import pycosa.modeling as modeling

Literal = Tuple[str, bool]


class CoverageResult(NamedTuple):
    """
    Result of a coverage measurement.

    Fields
    ------
    t : int
        Interaction strength that was measured.
    covered : int
        Number of valid interactions covered by the sample.
    valid : int
        Number of valid interactions in total.
    ratio : float
        ``covered / valid``; NaN if there are no valid interactions.
    missing : list of tuples, optional
        Valid but uncovered interactions (only with ``return_missing``).
    """

    t: int
    covered: int
    valid: int
    ratio: float
    missing: Optional[List[Tuple[Literal, ...]]]


def _align(vm: modeling.VariabilityModel, sample: pd.DataFrame) -> pd.DataFrame:
    """Reindexes the sample onto the model's options, in model order."""
    features = vm.get_binary_features()
    missing = [f for f in features if f not in sample.columns]
    if missing:
        raise ValueError("Sample lacks columns for options: {}".format(missing[:10]))
    return sample[features].astype(bool)


def validity(vm: modeling.VariabilityModel, sample: pd.DataFrame) -> float:
    """Fraction of configurations in the sample that satisfy the model."""
    sample = _align(vm, sample)
    if len(sample) == 0:
        return float("nan")
    solver = z3.Solver()
    solver.add(vm.get_binary_clauses())
    variables = vm.get_vars()
    valid = 0
    for row in sample.itertuples(index=False):
        assumptions = [v if val else z3.Not(v) for v, val in zip(variables, row)]
        valid += solver.check(*assumptions) == z3.sat
    return valid / len(sample)


def uniqueness(sample: pd.DataFrame) -> float:
    """Fraction of distinct configurations in the sample."""
    if len(sample) == 0:
        return float("nan")
    return len(sample.drop_duplicates()) / len(sample)


def distance_distribution(sample: pd.DataFrame) -> pd.Series:
    """Histogram of the number of selected options per configuration."""
    return sample.astype(bool).sum(axis=1).value_counts().sort_index()


def hamming_distances(sample: pd.DataFrame) -> np.ndarray:
    """Matrix of pairwise Hamming distances between configurations."""
    X = sample.to_numpy(dtype=float)
    return X @ (1 - X).T + (1 - X) @ X.T


def diversity(sample: pd.DataFrame) -> Dict[str, float]:
    """
    Diversity of a sample based on Hamming distances, normalized by the number
    of options:

    - ``mean_distance``: average distance between two configurations
    - ``mean_nearest``: average distance of a configuration to its nearest
      neighbour (low values indicate clusters)
    """
    n, m = sample.shape
    if n < 2 or m == 0:
        return {"mean_distance": float("nan"), "mean_nearest": float("nan")}
    D = hamming_distances(sample) / m
    off_diagonal = D[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    return {
        "mean_distance": float(off_diagonal.mean()),
        "mean_nearest": float(off_diagonal.min(axis=1).mean()),
    }


def option_frequency_error(
    vm: modeling.VariabilityModel,
    sample: pd.DataFrame,
    population: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Mean absolute difference between how often each option is selected in the
    sample and in the population of all valid configurations. A uniform
    random sample has an expected error close to 0; solver-based samples are
    often biased towards (de)selecting options.

    The population frequencies are computed by exact model counting unless
    given explicitly.
    """
    sample = _align(vm, sample)
    if len(sample) == 0:
        return float("nan")
    population = vm.option_frequencies() if population is None else population
    expected = np.array([population[f] for f in sample.columns])
    observed = sample.to_numpy(dtype=float).mean(axis=0)
    return float(np.abs(observed - expected).mean())


def _polarities(polarity: str, t: int):
    """Returns the value patterns to consider for a given polarity."""
    if polarity == "both":
        return list(itertools.product([True, False], repeat=t))
    if polarity == "positive":
        return [(True,) * t]
    if polarity == "negative":
        return [(False,) * t]
    raise ValueError("polarity must be 'both', 'positive' or 'negative'")


def t_wise_coverage(
    vm: modeling.VariabilityModel,
    sample: pd.DataFrame,
    t: int = 2,
    polarity: str = "both",
    return_missing: bool = False,
) -> CoverageResult:
    """
    Fraction of valid t-wise interactions covered by the sample.

    An interaction is an assignment of ``t`` distinct options to values
    (e.g., ``A=True, B=False``). It is valid if some valid configuration
    contains it, and covered if some configuration of the sample contains it.

    Parameters
    ----------
    t : int
        Interaction strength.
    polarity : {"both", "positive", "negative"}
        Which value combinations to consider: all (default, the usual
        definition of t-wise coverage), only selected options (what
        ``CoverageBasedSampler`` guarantees by default), or only deselected
        options (``negative_wise=True``).
    return_missing : bool
        Also return the list of valid but uncovered interactions.
    """
    if t < 1:
        raise ValueError("t must be a positive integer, got {}".format(t))
    sample = _align(vm, sample)
    features = vm.get_binary_features()
    variables = vm.get_vars()
    n = len(features)

    # one bitmask over rows per literal: bit r is set iff row r contains it
    def masks(X: np.ndarray):
        """Bitmask over rows per literal: bit r is set iff row r contains
        the literal."""
        pos, neg = [], []
        for j in range(X.shape[1]):
            col = X[:, j]
            pos.append(int("".join("1" if b else "0" for b in col[::-1]) or "0", 2))
            neg.append(int("".join("0" if b else "1" for b in col[::-1]) or "0", 2))
        return pos, neg

    sample_pos, sample_neg = masks(sample.to_numpy(dtype=bool))

    # witnesses: valid configurations found while checking interactions;
    # every interaction they contain is valid without another solver call
    witness_pos = [0] * n
    witness_neg = [0] * n
    n_witnesses = 0

    # random decision phases make witnesses diverse, so that each one
    # validates many further interactions
    solver = z3.SolverFor("QF_FD")
    solver.set("sat.phase", "random")
    solver.set("random_seed", 0)
    solver.add(vm.get_binary_clauses())

    def add_witness(model):
        """Registers one valid configuration; every interaction it contains
        is known to be valid without further solver calls."""
        nonlocal n_witnesses
        bit = 1 << n_witnesses
        n_witnesses += 1
        for j, v in enumerate(variables):
            if z3.is_true(model.eval(v, model_completion=True)):
                witness_pos[j] |= bit
            else:
                witness_neg[j] |= bit

    # literals that cannot occur in any valid configuration
    infeasible_literals = set()
    for j, v in enumerate(variables):
        for value in (True, False):
            if solver.check(v if value else z3.Not(v)) == z3.unsat:
                infeasible_literals.add((j, value))

    patterns = _polarities(polarity, t)
    covered = valid = 0
    missing: List[Tuple[Literal, ...]] = []

    for combo in itertools.combinations(range(n), t):
        for values in patterns:
            if any((j, val) in infeasible_literals for j, val in zip(combo, values)):
                continue

            mask = -1
            for j, val in zip(combo, values):
                mask &= sample_pos[j] if val else sample_neg[j]
                if mask == 0:
                    break
            if mask != 0:
                covered += 1
                valid += 1
                continue

            wmask = -1
            for j, val in zip(combo, values):
                wmask &= witness_pos[j] if val else witness_neg[j]
                if wmask == 0:
                    break
            if wmask == 0:
                assumptions = [
                    variables[j] if val else z3.Not(variables[j])
                    for j, val in zip(combo, values)
                ]
                if solver.check(*assumptions) != z3.sat:
                    continue
                add_witness(solver.model())

            valid += 1
            if return_missing:
                missing.append(
                    tuple((features[j], val) for j, val in zip(combo, values))
                )

    ratio = covered / valid if valid > 0 else float("nan")
    return CoverageResult(t, covered, valid, ratio, missing if return_missing else None)


def summarize(
    vm: modeling.VariabilityModel,
    sample: pd.DataFrame,
    t: Tuple[int, ...] = (1, 2),
    polarity: str = "both",
    population: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Computes all metrics for one sample."""
    result: Dict[str, float] = {
        "size": float(len(sample)),
        "validity": validity(vm, sample),
        "uniqueness": uniqueness(sample),
    }
    for strength in t:
        result["coverage_t{}".format(strength)] = t_wise_coverage(
            vm, sample, t=strength, polarity=polarity
        ).ratio
    result.update(diversity(sample))
    result["frequency_error"] = option_frequency_error(vm, sample, population)
    return result


def compare(
    vm: modeling.VariabilityModel,
    samples: Mapping[str, pd.DataFrame],
    t: Tuple[int, ...] = (1, 2),
    polarity: str = "both",
) -> pd.DataFrame:
    """
    Computes all metrics for several samples (e.g., of different sampling
    strategies) and returns them as a data frame with one row per sample.
    """
    population = vm.option_frequencies()
    rows = {
        name: summarize(vm, df, t=t, polarity=polarity, population=population)
        for name, df in samples.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


if __name__ == "__main__":
    pass
