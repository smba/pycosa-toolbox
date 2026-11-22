from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import z3


class VariabilityModel:
    """
    Propositional variability model of binary configuration options.

    The model is kept in two representations: a list of DIMACS-style clauses
    (lists of signed integers) and the equivalent list of z3 constraints.
    """

    def __init__(self):
        self.binary_options: List[str] = []

        self.dimacs: Optional[List[List[int]]] = None
        self.binary_clauses: Optional[List[z3.BoolRef]] = None
        self.bin_feature_to_index: Optional[Dict[str, int]] = None
        self.bin_index_to_feature: Optional[Dict[int, str]] = None

        self._vars: Dict[str, z3.BoolRef] = {}

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dimacs_file(cls, path: str) -> "VariabilityModel":
        """Convenience constructor that parses a DIMACS file."""
        from pycosa.parsing import DimacsParser

        parser = DimacsParser()
        parser.parse(path)
        vm = cls()
        vm.from_dimacs(parser.get_clauses(), parser.get_index2features())
        return vm

    def from_dimacs(
        self, dimacs: Sequence[Sequence[int]], index2features: Mapping[int, str]
    ) -> None:
        index2features = dict(index2features)

        # make sure that every variable referenced by a clause has a name
        for clause in dimacs:
            for literal in clause:
                index = abs(int(literal))
                if index not in index2features:
                    index2features[index] = "_x{}".format(index)

        self.bin_index_to_feature = dict(sorted(index2features.items()))
        self.bin_feature_to_index = {v: k for k, v in self.bin_index_to_feature.items()}
        self.binary_options = list(self.bin_index_to_feature.values())
        self._vars = {name: z3.Bool(name) for name in self.binary_options}

        # copy, so that later modifications do not leak into the caller's data
        self.dimacs = [[int(lit) for lit in clause] for clause in dimacs]
        self.binary_clauses = self._generate_z3_clauses(self.dimacs)

    # ------------------------------------------------------------------ #
    # z3 helpers
    # ------------------------------------------------------------------ #

    def var(self, feature: str) -> z3.BoolRef:
        """Returns the (cached) z3 variable of a feature."""
        return self._vars[feature]

    def get_vars(self) -> List[z3.BoolRef]:
        """Returns the z3 variables of all binary features in feature order."""
        return [self._vars[f] for f in self.binary_options]

    def _literal(self, literal: int, suffix: str = "") -> z3.BoolRef:
        name = self.bin_index_to_feature[abs(literal)]
        var = self._vars[name] if suffix == "" else z3.Bool(name + suffix)
        return var if literal > 0 else z3.Not(var)

    def _generate_z3_clauses(
        self, clauses: Sequence[Sequence[int]], suffix: str = ""
    ) -> List[z3.BoolRef]:
        z3_clauses = []
        for clause in clauses:
            ors = [self._literal(literal, suffix) for literal in clause]
            if len(ors) == 0:
                # the empty clause is unsatisfiable
                z3_clauses.append(z3.BoolVal(False))
            elif len(ors) == 1:
                z3_clauses.append(ors[0])
            else:
                z3_clauses.append(z3.Or(ors))

        return z3_clauses

    def get_binary_clauses(self) -> List[z3.BoolRef]:
        return self.binary_clauses

    def get_binary_features(self) -> List[str]:
        return list(self.binary_options)

    def get_z3_pair_clauses(self) -> List[z3.BoolRef]:
        """
        Returns the constraints of two independent copies of the model. The
        variables of the copies are suffixed with '-1' and '-2', respectively.
        """
        return self._generate_z3_clauses(
            self.dimacs, suffix="-1"
        ) + self._generate_z3_clauses(self.dimacs, suffix="-2")

    # ------------------------------------------------------------------ #
    # analysis
    # ------------------------------------------------------------------ #

    def is_valid(self, configuration: Mapping[str, bool]) -> bool:
        """
        Checks whether a (partial) configuration satisfies the model.
        Features missing from the configuration are left unconstrained.
        """
        solver = z3.Solver()
        solver.add(self.binary_clauses)
        assumptions = [
            self._vars[f] if bool(v) else z3.Not(self._vars[f])
            for f, v in configuration.items()
        ]
        return solver.check(*assumptions) == z3.sat

    def is_satisfiable(self) -> bool:
        return self.is_valid({})

    def core_and_dead_features(self):
        """
        Returns a tuple (core, dead) of features that are selected in every
        valid configuration (core) or in none (dead).
        """
        solver = z3.Solver()
        solver.add(self.binary_clauses)
        core, dead = [], []
        for feature in self.binary_options:
            v = self._vars[feature]
            if solver.check(z3.Not(v)) == z3.unsat:
                core.append(feature)
            elif solver.check(v) == z3.unsat:
                dead.append(feature)
        return core, dead

    # ------------------------------------------------------------------ #
    # modification
    # ------------------------------------------------------------------ #

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffles the order of clauses and of literals within each clause."""
        rng = np.random.default_rng(seed)
        clauses = [
            [clause[i] for i in rng.permutation(len(clause))] for clause in self.dimacs
        ]
        self.dimacs = [clauses[i] for i in rng.permutation(len(clauses))]
        self.binary_clauses = self._generate_z3_clauses(self.dimacs)

    def _add_unit_clauses(self, options: Sequence[str], positive: bool) -> None:
        if isinstance(options, str):
            options = [options]
        sign = 1 if positive else -1
        for opt in options:
            self.dimacs.append([sign * self.bin_feature_to_index[opt]])
        self.binary_clauses = self._generate_z3_clauses(self.dimacs)

    def select(self, options: Sequence[str]) -> None:
        """Forces the given options to be selected in every configuration."""
        self._add_unit_clauses(options, positive=True)

    def deselect(self, options: Sequence[str]) -> None:
        """Forces the given options to be deselected in every configuration."""
        self._add_unit_clauses(options, positive=False)


if __name__ == "__main__":
    pass
