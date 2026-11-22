"""
Sampling strategies for configurations of a variability model.

All model-based samplers share the same interface: they take a
:class:`pycosa.modeling.VariabilityModel` (plus an optional seed for
reproducibility), and :meth:`sample <_Sampler.sample>` returns a
``pandas.DataFrame`` with one boolean column per option and one row per
configuration.

Available strategies:

- :class:`DFSSampler` enumerates distinct valid configurations, spread out
  via randomized solver phases.
- :class:`DistanceBasedSampler` samples distances (number of selected
  options) uniformly and finds a valid configuration for each distance.
- :class:`UniformSampler` draws every valid configuration with exactly the
  same probability (via exact model counting).
- :class:`CoverageBasedSampler` greedily covers all t-wise interactions.
- :class:`GroupSampler` draws pairs of configurations that differ exactly
  in a given group of options (for elementary-effect measurements).
- :class:`OfflineSampler` finds pairs in third-party data sets.

Metrics for comparing the resulting samples are provided by
:mod:`pycosa.metrics`.
"""

import itertools
import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import z3

import pycosa.modeling as modeling


def _new_solver(seed: Optional[int] = None, random_phase: bool = True) -> z3.Solver:
    """
    Creates a solver for finite-domain (propositional + pseudo-Boolean)
    problems. With ``random_phase``, the SAT core picks random polarities for
    decisions, which yields far more diverse solutions than the default
    (which always prefers 'False' and thus clusters around minimal configs).
    """
    solver = z3.SolverFor("QF_FD")
    if seed is not None:
        solver.set("random_seed", int(seed) % (2**31))
    if random_phase:
        solver.set("sat.phase", "random")
    return solver


class _Sampler:
    """
    Base class of the model-based samplers.

    Holds the model, its z3 variables, and helpers shared by all sampling
    strategies: solver construction, translating solver models into
    configurations, blocking clauses, and building the result data frame.
    Subclasses must implement :meth:`sample`.
    """

    def __init__(self, vm: modeling.VariabilityModel, seed: Optional[int] = None):
        """Snapshots the model's clauses and prepares reproducible randomness
        (``seed``) for subsequent sampling runs."""
        self.vm = vm
        self.vm = vm
        self.clauses = vm.get_binary_clauses()
        self.bin_features = vm.get_binary_features()
        self.vars = vm.get_vars()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int = 10) -> pd.DataFrame:
        """Draws ``size`` configurations and returns them as a data frame with
        one boolean column per option; implemented by subclasses."""
        raise NotImplementedError()

    def _solver(self, random_phase: bool = True) -> z3.Solver:
        """Returns a solver pre-loaded with the model's clauses."""
        seed = int(self.rng.integers(0, 2**31 - 1))
        solver = _new_solver(seed=seed, random_phase=random_phase)
        solver.add(self.clauses)
        return solver

    def _model_to_config(self, model: z3.ModelRef, variables=None) -> List[bool]:
        """Transforms a z3 model into a list of booleans, in the order of the
        given (or the model's) variables; unassigned variables are completed."""
        variables = self.vars if variables is None else variables
        return [z3.is_true(model.eval(v, model_completion=True)) for v in variables]

    @staticmethod
    def _blocking_clause(variables, config: Sequence[bool]) -> z3.BoolRef:
        """Clause that excludes exactly the given assignment of variables."""
        return z3.Or([z3.Not(v) if val else v for v, val in zip(variables, config)])

    def _create_uniqueness_constraint(self, model: z3.ModelRef) -> z3.BoolRef:
        """Returns the blocking clause excluding exactly the configuration
        described by the given z3 model."""
        return self._blocking_clause(self.vars, self._model_to_config(model))

    def _to_frame(self, configs: Sequence[Sequence[bool]]) -> pd.DataFrame:
        """Builds the result data frame with one boolean column per option."""
        return pd.DataFrame(list(configs), columns=self.bin_features, dtype=bool)

    def _warn_short(self, got: int, wanted: int) -> None:
        """Logs a warning if fewer configurations were found than requested."""
        if got < wanted:
            logging.warning(
                "%s: only %d of %d requested configurations could be found.",
                self.__class__.__name__,
                got,
                wanted,
            )


class DFSSampler(_Sampler):
    """
    Enumerates distinct valid configurations by repeatedly asking the solver
    for a new solution. By default the solver uses random decision phases, so
    the sample is spread across the configuration space. Pass
    ``random_phase=False`` to obtain the classic (clustered) enumeration.
    """

    def __init__(self, vm, seed: Optional[int] = None, random_phase: bool = True):
        """Parameters are as described in the class docstring."""
        super().__init__(vm, seed)
        self.random_phase = random_phase

    def sample(self, size: int = 10) -> pd.DataFrame:
        """Parameters are as described in the class docstring."""
        solver = self._solver(random_phase=self.random_phase)

        solutions = []
        while len(solutions) < size and solver.check() == z3.sat:
            config = self._model_to_config(solver.model())
            solutions.append(config)
            solver.add(self._blocking_clause(self.vars, config))

        self._warn_short(len(solutions), size)
        return self._to_frame(solutions)


class CoverageBasedSampler(_Sampler):
    """
    Greedy t-wise coverage sampling. Every valid interaction of ``t`` options
    appears in at least one configuration of the sample. Each new
    configuration covers a previously uncovered interaction and is greedily
    extended with further uncovered interactions, which keeps the sample
    small.

    Which interactions are covered depends on ``polarity``: combinations of
    selected options (``"positive"``, the default), of deselected options
    (``"negative"``), or all value combinations (``"both"``, the usual
    definition of t-wise coverage in combinatorial interaction testing).
    """

    def __init__(self, vm, seed: Optional[int] = None):
        """Parameters are as described in the class docstring."""
        super().__init__(vm, seed)

    def _combinations(
        self, t: int, polarity: str
    ) -> List[Tuple[Tuple[int, bool], ...]]:
        """Returns all index/value patterns to cover, as tuples of
        (variable index, value) pairs."""
        if polarity == "positive":
            patterns = [(True,) * t]
        elif polarity == "negative":
            patterns = [(False,) * t]
        elif polarity == "both":
            patterns = list(itertools.product([True, False], repeat=t))
        else:
            raise ValueError("polarity must be 'positive', 'negative' or 'both'")
        idx = list(range(len(self.vars)))
        return [
            tuple(zip(combo, values))
            for combo in itertools.combinations(idx, t)
            for values in patterns
        ]

    def sample(
        self,
        t: int = 1,
        negative_wise: bool = False,
        extend: bool = True,
        max_tries: Optional[int] = None,
        polarity: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        t : int
            Interaction strength (1 = feature-wise, 2 = pair-wise, ...).
        negative_wise : bool
            Shorthand for ``polarity="negative"``.
        polarity : {"positive", "negative", "both"}, optional
            Which interactions to cover (see class docstring). Defaults to
            ``"negative"`` if ``negative_wise`` else ``"positive"``.
        extend : bool
            Greedily pack further uncovered combinations into each
            configuration (smaller samples, more solver calls).
        max_tries : int, optional
            Upper bound on solver calls per configuration during extension.
        """
        if t < 1:
            raise ValueError("t must be a positive integer, got {}".format(t))

        solver = self._solver(random_phase=False)

        if polarity is None:
            polarity = "negative" if negative_wise else "positive"
        elif negative_wise and polarity != "negative":
            raise ValueError(
                "negative_wise=True contradicts polarity={!r}".format(polarity)
            )

        combos = self._combinations(t, polarity)
        order = self.rng.permutation(len(combos))
        uncovered = [combos[i] for i in order]

        # cached z3 literals: passing few, pre-built assumptions per check
        # avoids most of the overhead of z3's Python bindings
        lits = {}
        for i, v in enumerate(self.vars):
            lits[(i, True)] = v
            lits[(i, False)] = z3.Not(v)

        solutions: List[List[bool]] = []
        while uncovered:
            target = uncovered[0]
            uncovered = uncovered[1:]
            if solver.check(*[lits[lit] for lit in target]) != z3.sat:
                continue  # infeasible interaction

            if extend:
                # greedily try to include further uncovered combinations;
                # accepted literals are added permanently within a scope
                config = self._model_to_config(solver.model())
                solver.push()
                solver.add(*[lits[lit] for lit in target])
                fixed = dict(target)
                tries = 0
                for combo in uncovered:
                    if max_tries is not None and tries >= max_tries:
                        break
                    if all(config[i] == val for i, val in combo):
                        continue
                    # skip combinations that contradict already fixed literals
                    if any(fixed.get(i, val) != val for i, val in combo):
                        continue
                    tries += 1
                    new = [lits[lit] for lit in combo if lit[0] not in fixed]
                    if solver.check(*new) == z3.sat:
                        config = self._model_to_config(solver.model())
                        solver.add(*new)
                        fixed.update(combo)
                solver.pop()
            else:
                config = self._model_to_config(solver.model())

            solutions.append(config)
            uncovered = [
                c for c in uncovered if not all(config[i] == val for i, val in c)
            ]

        return self._to_frame(solutions)


class DistanceBasedSampler(_Sampler):
    """
    Distance-based sampling (Kaltenecker et al., ICSE 2019): for each
    configuration, a distance ``d`` (number of selected options) is drawn
    uniformly from the feasible distances, and a valid configuration with
    exactly ``d`` selected options is searched.
    """

    def __init__(self, vm: modeling.VariabilityModel, seed: Optional[int] = None):
        """Parameters are as described in the class docstring."""
        super().__init__(vm, seed)

    def sample(self, size: int = 10) -> pd.DataFrame:
        """Parameters are as described in the class docstring."""
        n_options = len(self.vars)
        weighted = [(v, 1) for v in self.vars]

        solvers: Dict[int, z3.Solver] = {}
        available = list(range(n_options + 1))

        solutions = []
        while len(solutions) < size and available:
            d = int(self.rng.choice(available))

            if d not in solvers:
                solvers[d] = self._solver()
                solvers[d].add(z3.PbEq(weighted, d))
            solver = solvers[d]

            if solver.check() == z3.sat:
                config = self._model_to_config(solver.model())
                solutions.append(config)
                # block this configuration in its distance solver; other
                # distance solvers cannot produce it anyway
                solver.add(self._blocking_clause(self.vars, config))
            else:
                available.remove(d)
                del solvers[d]

        self._warn_short(len(solutions), size)
        return self._to_frame(solutions)


class UniformSampler(_Sampler):
    """
    Uniform random sampling: every valid configuration is drawn with exactly
    the same probability. Uses exact model counting (``pycosa.counting``), so
    no solver randomness or rejection is involved.

    Uniform samples are the usual baseline for comparing sampling strategies
    and are unbiased w.r.t. the configuration space (unlike solver-based
    enumeration, which tends to over-represent some regions).
    """

    def __init__(self, vm: modeling.VariabilityModel, seed: Optional[int] = None):
        """Computes the exact model counter upfront, which may take a
        moment on large models. The parameters are as described in the
        class docstring."""
        super().__init__(vm, seed)
        self.counter = vm.model_counter()
        self._py_rng = random.Random(seed)
        self._feature_indexes = [vm.bin_feature_to_index[f] for f in self.bin_features]

    def _draw(self) -> List[bool]:
        """Draws one uniformly distributed configuration (in option order)."""
        assignment = self.counter.sample(self._py_rng)
        return [assignment[i] for i in self._feature_indexes]

    def sample(
        self, size: int = 10, unique: bool = True, max_attempts: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        size : int
            Number of configurations.
        unique : bool
            Sample without replacement (default). Duplicates are rejected and
            redrawn, which keeps each set of distinct configurations equally
            likely. With ``unique=False``, configurations are i.i.d. draws.
        max_attempts : int, optional
            Upper bound on draws when sampling without replacement
            (default: ``20 * size + 100``).
        """
        total = self.counter.count()
        if total == 0:
            raise ValueError("The variability model is unsatisfiable.")

        if not unique:
            return self._to_frame([self._draw() for _ in range(size)])

        target = min(size, total)
        if target < size:
            logging.warning(
                "UniformSampler: the model has only %d valid configurations.", total
            )
        max_attempts = 20 * size + 100 if max_attempts is None else max_attempts

        seen, solutions = set(), []
        attempts = 0
        while len(solutions) < target and attempts < max_attempts:
            attempts += 1
            config = self._draw()
            key = tuple(config)
            if key not in seen:
                seen.add(key)
                solutions.append(config)

        self._warn_short(len(solutions), size)
        return self._to_frame(solutions)


class GroupSampler:
    """
    Samples pairs of configurations that differ exactly in a given group of
    options: in the first configuration of each pair, all group options are
    selected, in the second all are deselected; all other options agree.
    """

    def __init__(self, vm: modeling.VariabilityModel, seed: Optional[int] = None):
        """Parameters are as described in the class docstring."""
        self.vm = vm
        self.bin_features = vm.get_binary_features()
        self.clauses = self.vm.get_z3_pair_clauses()
        self.rng = np.random.default_rng(seed)
        self.vars_en = [z3.Bool(f"{o}-1") for o in self.bin_features]
        self.vars_dis = [z3.Bool(f"{o}-2") for o in self.bin_features]

    def sample(
        self, options: Sequence[str], size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Draws at most ``size`` pairs; returns two data frames with one
        boolean column per option (options enabled vs. disabled in the group).
        """
        options = set(options)
        unknown = options - set(self.bin_features)
        if unknown:
            raise ValueError("Unknown options: {}".format(sorted(unknown)))

        base = _new_solver(seed=int(self.rng.integers(0, 2**31 - 1)))
        base.add(self.clauses)
        for option, en, dis in zip(self.bin_features, self.vars_en, self.vars_dis):
            if option in options:
                base.add(en, z3.Not(dis))
            else:
                base.add(en == dis)

        enabled, disabled = [], []
        n_options = len(self.bin_features)
        weighted = [(v, 1) for v in self.vars_en]
        available = list(range(n_options + 1))

        while len(enabled) < size and available:
            d = int(self.rng.choice(available))
            if base.check(z3.PbEq(weighted, d)) != z3.sat:
                available.remove(d)
                continue

            model = base.model()
            en = [
                z3.is_true(model.eval(v, model_completion=True)) for v in self.vars_en
            ]
            dis = [
                z3.is_true(model.eval(v, model_completion=True)) for v in self.vars_dis
            ]
            enabled.append(en)
            disabled.append(dis)

            # the disabled configuration is determined by the enabled one, so
            # blocking the enabled one suffices for uniqueness of pairs
            base.add(_Sampler._blocking_clause(self.vars_en, en))

        if len(enabled) < size:
            logging.warning(
                "GroupSampler: only %d of %d requested pairs could be found.",
                len(enabled),
                size,
            )

        enabled = pd.DataFrame(enabled, columns=self.bin_features, dtype=bool)
        disabled = pd.DataFrame(disabled, columns=self.bin_features, dtype=bool)

        return enabled, disabled


class OfflineSampler:
    """
    Wrapper class for sampling strategies on already existing data. The intended use case
    are third-party data sets.
    """

    def __init__(self, df: pd.DataFrame, shuffle=False, seed: Optional[int] = None):
        """Wraps the given data frame; with ``shuffle``, rows are put into a
        random order (reproducible via ``seed``)."""
        if shuffle:
            df = df.sample(frac=1, random_state=seed)

        self.df = df

    def elementary_effect_sample(self, options: Sequence[object], size: int = 100):
        """
        Finds pairs of configurations in the data set that differ only in the
        given options (all selected vs. all deselected).

        Returns two lists of row indexes (enabled, disabled) of equal length.
        """
        options = list(options)
        df = self.df

        selected = df[options].astype(bool).all(axis=1)
        deselected = ~df[options].astype(bool).any(axis=1)
        rest = df.drop(columns=options)

        # rows indexed by the values of all remaining columns
        key = pd.MultiIndex.from_frame(rest) if rest.shape[1] > 0 else None
        if key is None:
            en_rows = list(df.index[selected])[:1]
            dis_rows = list(df.index[deselected])[:1]
            n = min(len(en_rows), len(dis_rows), size)
            return en_rows[:n], dis_rows[:n]

        first_disabled = {}
        for row, k, is_dis in zip(df.index, key, deselected):
            if is_dis and k not in first_disabled:
                first_disabled[k] = row

        en, dis, used = [], [], set()
        for row, k, is_sel in zip(df.index, key, selected):
            if len(en) >= size:
                break
            if is_sel and k in first_disabled and k not in used:
                used.add(k)
                en.append(row)
                dis.append(first_disabled[k])

        return en, dis


if __name__ == "__main__":
    pass
