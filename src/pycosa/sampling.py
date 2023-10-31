from typing import Sequence

import pandas as pd
import z3

import pycosa.modeling as modeling
import itertools
import numpy as np


class _Sampler:
    def __init__(self, vm: modeling.VariabilityModel):
        self.clauses = vm.get_binary_clauses()
        self.bin_features = vm.get_binary_features()

    def sample(self, size: int = 10) -> pd.DataFrame:
        pass

    def _create_uniqueness_constraint(self, model: z3.ModelRef) -> z3.Or:
        literals = [z3.Bool(literal) for literal in self.bin_features]
        exclusion_constraint = z3.Or(
            [p != model.evaluate(p, model_completion=True) for p in literals]
        )
        return exclusion_constraint


class DFSSampler(_Sampler):
    def __init__(self, vm):
        super().__init__(vm)

    def sample(self, size: int = 10) -> pd.DataFrame:
        solver = z3.Solver()
        solver.add(self.clauses)

        solutions = []
        for i in range(size):

            if solver.check() == z3.sat:
                model = solver.model()

                # construct solution
                solution = {}
                for literal in self.bin_features:
                    val = model[z3.Bool(literal)]
                    if val is not None:
                        solution[literal] = val.__bool__()
                    else:
                        solution[literal] = False

                solutions.append(solution)

                # add uniqueness constraint
                uniqueness_constraint = self._create_uniqueness_constraint(model)
                solver.add(uniqueness_constraint)

        df = pd.DataFrame(solutions)
        return df

class CoverageBasedSampler(_Sampler):
    def __init__(self, vm):
        super().__init__(vm)

    def sample(self, t = 1, negative_wise = False) -> pd.DataFrame:
        solver = z3.Solver()
        solver.add(self.clauses)

        solutions = []
        assert t > 0
        for term in itertools.combinations(self.bin_features, t):
            solver.push()
            
            # construct constraints
            for option in term:
                if not negative_wise:
                    solver.add(z3.Bool(option))
                else:
                    solver.add(z3.Not(z3.Bool(option)))
            
            if solver.check() == z3.sat:
                # keep record of solution
                solution = {}
                for literal in self.bin_features:
                    val = model[z3.Bool(literal)]
                    if val is not None:
                        solution[literal] = val.__bool__()
                    else:
                        solution[literal] = False
                solutions.append(solution)
                
            solver.pop()
            
            # add uniqueness constraint
            uniqueness_constraint = self._create_uniqueness_constraint(model)
            solver.add(uniqueness_constraint)
        
        df = pd.DataFrame(solutions)
        return df
        
        
        for i in range(size):

            if solver.check() == z3.sat:
                model = solver.model()

                # keep record of solution
                solution = {}
                for literal in self.bin_features:
                    val = model[z3.Bool(literal)]
                    if val is not None:
                        solution[literal] = val.__bool__()
                    else:
                        solution[literal] = False

                solutions.append(solution)

                # add uniqueness constraint
                uniqueness_constraint = self._create_uniqueness_constraint(model)
                solver.add(uniqueness_constraint)

        df = pd.DataFrame(solutions)
        return df

class DistanceBasedSampler(_Sampler):
    def __init__(self, vm: modeling.VariabilityModel):
        super().__init__(vm)

    def sample(self, size: int = 10) -> pd.DataFrame:

        n_options = len(self.bin_features)

        # Construct some Solver instances
        solvers = {}
        for d in range(n_options):
            solvers[d] = z3.Solver()
            solvers[d].add(self.clauses)
            solvers[d].add(
                z3.Sum([z3.If(z3.Bool(opt), 1, 0) for opt in range(n_options)]) == d
            )

        solutions = []
        for i in range(size):

            d = np.random.choice(np.arange(n_options), size=1, replace=True)[0]
            d = int(d)

            if solvers[d].check() == z3.sat:
                model = solvers[d].model()

                # construct solution
                solution = {}
                for j in range(n_options):
                    option = self.bin_features[j]
                    val = model[z3.Bool(option)]
                    if val is not None:
                        solution[option] = val.__bool__()
                    else:
                        solution[option] = False

                solutions.append(solution)

                # add uniqueness constraint
                uniqueness_constraint = self._create_uniqueness_constraint(model)
                solvers[d].add(uniqueness_constraint)

        df = pd.DataFrame(solutions)
        return df


class GroupSampler:
    def __init__(self, vm: modeling.VariabilityModel):
        self.vm = vm
        self.bin_features = vm.get_binary_features()
        self.clauses = self.vm.get_z3_pair_clauses()

    def sample(self, options: Sequence[str], size: int):
        old_solutions = []

        enabled = []
        disabled = []

        n_options = len(self.bin_features)

        available_distances = set(list(range(n_options)))

        while len(enabled) < size and len(available_distances) > 0:
            solver = z3.Solver()

            solver.add(old_solutions)
            solver.add(self.clauses)

            # add distance constraint
            d = int(np.random.choice(list(available_distances), size=1)[0])
            solver.add(
                z3.Sum(
                    [
                        z3.If(z3.Bool(f"{option}-1"), 1, 0)
                        for option in self.bin_features
                    ]
                )
                == d
            )

            for option in self.bin_features:
                if option in options:

                    solver.add(
                        z3.And(z3.Bool(f"{option}-1"), z3.Not(z3.Bool(f"{option}-2")))
                    )
                else:
                    solver.add(z3.Bool(f"{option}-1") == z3.Bool(f"{option}-2"))

            if solver.check() == z3.sat:
                model = solver.model()

                # construct solutions
                en_solution = {}
                dis_solution = {}
                for option in self.bin_features:
                    en_val = model[z3.Bool(f"{option}-1")]
                    dis_val = model[z3.Bool(f"{option}-2")]

                    if en_val is not None and dis_val is not None:
                        en_solution[f"{option}"] = en_val.__bool__()
                        dis_solution[f"{option}"] = dis_val.__bool__()
                    else:
                        en_solution[f"{option}"] = False
                        dis_solution[f"{option}"] = False

                enabled.append(en_solution)
                disabled.append(dis_solution)

                # add uniqueness constraint
                en_literals = [z3.Bool(f"{option}-1") for option in self.bin_features]
                dis_literals = [z3.Bool(f"{option}-2") for option in self.bin_features]

                old_solutions.append(
                    z3.Or(
                        [
                            p != model.evaluate(p, model_completion=True)
                            for p in en_literals
                        ]
                    )
                )
                old_solutions.append(
                    z3.Or(
                        [
                            p != model.evaluate(p, model_completion=True)
                            for p in dis_literals
                        ]
                    )
                )

            else:
                available_distances = available_distances - {d}

        enabled = pd.DataFrame(enabled)
        disabled = pd.DataFrame(disabled)

        return enabled, disabled


class OfflineSampler:
    """
    Wrapper class for sampling strategies on already existing data. The intended use case
    are third-party data sets.
    """

    def __init__(self, df: pd.DataFrame, shuffle=False):

        if shuffle:
            df = df.sample(frac=1)

        self.df = df

    def elementary_effect_sample(self, options: Sequence[object], size: int = 100):

        df = self.df.copy()

        df["selected"] = df[options].all(axis=1)
        df["deselected"] = ~df[options].any(axis=1)

        drop_cols = options + ["selected", "deselected"]
        enabled = df[df["selected"]].drop(columns=drop_cols)
        disabled = df[df["deselected"]].drop(columns=drop_cols)

        configs = pd.concat([enabled, disabled])
        dups = configs[configs.duplicated(keep=False)]

        en, dis = [], []
        for rand, pair in dups.groupby(by=list(dups.columns)):
            pair["selected"] = df.loc[pair.index]["selected"]
            pair = pair.sort_values(by="selected")
            if pair.shape[0] == 2 and len(en) < size:
                en.append(pair.index[0])
                dis.append(pair.index[1])

        return en, dis


if __name__ == "__main__":
    pass
