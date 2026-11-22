"""
Exact model counting and uniform sampling for propositional (CNF) models.

The counter is a DPLL-style #SAT solver with unit propagation, decomposition
into independent components, and component caching. Uniform sampling follows
the same decision structure: at each decision, a branch is taken with
probability proportional to the number of solutions below it, so every valid
configuration is drawn with exactly the same probability.

This works well for typical feature models (hundreds of options); like every
exact approach, it can become slow for very large or highly constrained
models.
"""

import random
import sys
from collections import Counter as _Counter
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

Clause = Tuple[int, ...]
Formula = FrozenSet[Clause]


def _assign(clauses: Iterable[Clause], literal: int) -> Optional[Formula]:
    """Conditions the formula on ``literal``. Returns None on conflict."""
    out = []
    for clause in clauses:
        if literal in clause:
            continue
        if -literal in clause:
            clause = tuple(lit for lit in clause if lit != -literal)
            if len(clause) == 0:
                return None
        out.append(clause)
    return frozenset(out)


def _propagate(clauses: Formula) -> Tuple[Optional[Formula], List[int]]:
    """Unit propagation. Returns (formula, implied literals) or (None, ...)."""
    implied: List[int] = []
    while True:
        unit = next((c[0] for c in clauses if len(c) == 1), None)
        if unit is None:
            return clauses, implied
        implied.append(unit)
        clauses = _assign(clauses, unit)
        if clauses is None:
            return None, implied


def _variables(clauses: Iterable[Clause]) -> FrozenSet[int]:
    return frozenset(abs(lit) for clause in clauses for lit in clause)


def _components(clauses: Formula) -> List[Formula]:
    """Splits a formula into variable-disjoint components (union-find)."""
    parent: Dict[int, int] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for clause in clauses:
        for lit in clause:
            parent.setdefault(abs(lit), abs(lit))
        root = find(abs(clause[0]))
        for lit in clause[1:]:
            other = find(abs(lit))
            if other != root:
                parent[other] = root

    groups: Dict[int, List[Clause]] = {}
    for clause in clauses:
        groups.setdefault(find(abs(clause[0])), []).append(clause)
    return [frozenset(g) for g in groups.values()]


def _branch_variable(component: Formula) -> int:
    """Picks the variable occurring most often, preferring short clauses."""
    scores: _Counter = _Counter()
    for clause in component:
        weight = 1.0 / len(clause)
        for lit in clause:
            scores[abs(lit)] += weight
    # deterministic tie-breaking keeps the cache effective
    return max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]


class ModelCounter:
    """
    Exact model counter and uniform sampler for a CNF over a fixed set of
    variables.

    Parameters
    ----------
    clauses : sequence of sequences of int
        DIMACS-style clauses.
    variables : iterable of int, optional
        All variables of the model (variables without clauses are free).
        Defaults to the variables occurring in the clauses.
    """

    def __init__(
        self,
        clauses: Sequence[Sequence[int]],
        variables: Optional[Iterable[int]] = None,
    ):
        self.formula: Formula = frozenset(
            tuple(sorted(set(int(lit) for lit in clause))) for clause in clauses
        )
        # tautological clauses (x | -x) are always satisfied
        self.formula = frozenset(
            c for c in self.formula if not any(-lit in c for lit in c)
        )
        vars_in_clauses = _variables(self.formula)
        self.variables: FrozenSet[int] = frozenset(
            vars_in_clauses if variables is None else set(variables)
        )
        if not vars_in_clauses <= self.variables:
            missing = sorted(vars_in_clauses - self.variables)
            raise ValueError("Clauses use undeclared variables: {}".format(missing))

        self._cache: Dict[Formula, int] = {}
        # recursion depth grows with the number of decisions
        needed = 4 * len(self.variables) + 1000
        if sys.getrecursionlimit() < needed:
            sys.setrecursionlimit(needed)

    # ------------------------------------------------------------------ #
    # counting
    # ------------------------------------------------------------------ #

    def _count(self, clauses: Optional[Formula], variables: FrozenSet[int]) -> int:
        if clauses is None:
            return 0
        clauses, implied = _propagate(clauses)
        if clauses is None:
            return 0
        variables = variables - {abs(lit) for lit in implied}
        if len(clauses) == 0:
            return 2 ** len(variables)

        free = variables - _variables(clauses)
        result = 2 ** len(free)
        for component in _components(clauses):
            c = self._count_component(component)
            if c == 0:
                return 0
            result *= c
        return result

    def _count_component(self, component: Formula) -> int:
        cached = self._cache.get(component)
        if cached is not None:
            return cached
        variables = _variables(component)
        v = _branch_variable(component)
        rest = variables - {v}
        count = self._count(_assign(component, v), rest) + self._count(
            _assign(component, -v), rest
        )
        self._cache[component] = count
        return count

    def count(self, assumptions: Sequence[int] = ()) -> int:
        """Number of solutions, optionally under a list of assumed literals."""
        clauses: Optional[Formula] = self.formula
        variables = self.variables
        for lit in assumptions:
            if abs(lit) not in self.variables:
                raise ValueError("Unknown variable {}".format(abs(lit)))
            if clauses is None:
                return 0
            clauses = _assign(clauses, lit)
            variables = variables - {abs(lit)}
        return self._count(clauses, variables)

    def marginals(self) -> Dict[int, float]:
        """Fraction of solutions in which each variable is true."""
        total = self.count()
        if total == 0:
            raise ValueError("The formula is unsatisfiable.")
        return {v: self.count([v]) / total for v in sorted(self.variables)}

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #

    def _sample(
        self,
        clauses: Formula,
        variables: FrozenSet[int],
        rng: random.Random,
        out: Dict[int, bool],
    ) -> None:
        clauses, implied = _propagate(clauses)
        assert clauses is not None, "sampling from an unsatisfiable formula"
        for lit in implied:
            out[abs(lit)] = lit > 0
        variables = variables - {abs(lit) for lit in implied}

        for v in variables - _variables(clauses):
            out[v] = rng.random() < 0.5

        for component in _components(clauses):
            v = _branch_variable(component)
            rest = _variables(component) - {v}
            pos = _assign(component, v)
            neg = _assign(component, -v)
            n_pos = self._count(pos, rest)
            n_neg = self._count(neg, rest)
            if rng.randrange(n_pos + n_neg) < n_pos:
                out[v] = True
                self._sample(pos, rest, rng, out)
            else:
                out[v] = False
                self._sample(neg, rest, rng, out)

    def sample(self, rng: Optional[random.Random] = None) -> Dict[int, bool]:
        """Draws one solution uniformly at random."""
        rng = rng if rng is not None else random.Random()
        if self.count() == 0:
            raise ValueError("The formula is unsatisfiable.")
        out: Dict[int, bool] = {}
        self._sample(self.formula, self.variables, rng, out)
        return out


if __name__ == "__main__":
    pass
