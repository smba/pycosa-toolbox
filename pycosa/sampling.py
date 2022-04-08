import itertools
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
import z3

import pycosa.modeling as modeling


def int_to_config(i: int, n_options: int) -> np.ndarray:
    without_offset = np.array([int(x) for x in np.binary_repr(i)])
    offset = n_options + 1 - len(without_offset)
    binary = np.append(np.zeros(dtype=int, shape=offset), without_offset)

    return np.array(list(reversed(binary)))


class Sampler(ABC):
    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        self.fm = fm
        seed = kwargs.get("seed", 1)
        np.random.seed(seed)


class MultiSampler(Sampler):
    """
    Abstract wrapper to distinguish between sampling strategies that
    yielf one / or more than one configuration per iteration (like this).
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    @abstractmethod
    def sample(self, **kwargs) -> pd.DataFrame:
        pass


class SingleSampler(Sampler):
    """
    Abstract wrapper to distinguish between sampling strategies that
    yielf one (like this) / or more than one configuration per iteration.

    The method provided by this wrapper are only intendd to be used when
    using guided sampling, i.e., sampling from a subset of the entire
    variability/feature model. In addition, you can use these methods as a
    shortcut to depict non-trivial constraints, such as "of these 8 options, no more than
    4 can be enabled at the same time".
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        self.side_constraints = []
        super().__init__(fm, **kwargs)

    @abstractmethod
    def sample(self, **kwargs) -> pd.DataFrame:
        pass

    def constrain_enabled(self, options):
        """
        Specify that certain options need to be enabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        for opt_id in option_ids:
            self.side_constraints.append(
                z3.Extract(opt_id, opt_id, self.fm.target) == 1
            )

    def constrain_disabled(self, options):
        """
        Specify that certain options need to be disabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        for opt_id in option_ids:
            self.side_constraints.append(
                z3.Extract(opt_id, opt_id, self.fm.target) == 0
            )

    def constrain_min_enabled(self, options, minimum: int):
        """
        Specify that a minimum number of certain options needs to be enabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.
        minimum : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        n_options = len(self.fm.feature_map)

        self.side_constraints.append(
            z3.Sum(
                [
                    z3.ZeroExt(
                        n_options + 1, z3.Extract(opt_id, opt_id, self.fm.target)
                    )
                    for opt_id in option_ids
                ]
            )
            >= minimum
        )

    def constrain_max_enabled(self, options, maximum: int):
        """
        Specify that a maximum number of certain options needs to be enabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.
        maximum : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        n_options = len(self.fm.feature_map)

        self.side_constraints.append(
            z3.Sum(
                [
                    z3.ZeroExt(
                        n_options + 1, z3.Extract(opt_id, opt_id, self.fm.target)
                    )
                    for opt_id in option_ids
                ]
            )
            <= maximum
        )

    def constrain_min_disabled(self, options, minimum: int):
        """
        Specify that a minimum number of certain options needs to be disabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.
        minimum : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        n_options = len(self.fm.feature_map)

        self.side_constraints.append(
            z3.Sum(
                [
                    z3.ZeroExt(
                        n_options + 1, z3.Extract(opt_id, opt_id, self.fm.target)
                    )
                    for opt_id in option_ids
                ]
            )
            <= len(options) - minimum
        )

    def constrain_max_disabled(self, options, maximum: int):
        """
        Specify that a maximum number of certain options needs to be disabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.
        maximum : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        n_options = len(self.fm.feature_map)

        self.side_constraints.append(
            z3.Sum(
                [
                    z3.ZeroExt(
                        n_options + 1, z3.Extract(opt_id, opt_id, self.fm.target)
                    )
                    for opt_id in option_ids
                ]
            )
            >= len(options) - maximum
        )

    def constrain_subset_enabled(self, options, n: int):
        """
        Specify that a specific number of certain options needs to be enabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.
        n : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        n_options = len(self.fm.feature_map)

        self.side_constraints.append(
            z3.Sum(
                [
                    z3.ZeroExt(
                        n_options + 1, z3.Extract(opt_id, opt_id, self.fm.target)
                    )
                    for opt_id in option_ids
                ]
            )
            == n
        )

    def constrain_subset_disabled(self, options, n: int):
        """
        Specify that a specific number of certain options needs to be disabled.

        Parameters
        ----------
        options : TYPE
            DESCRIPTION.
        n : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        option_ids = [self.fm.feature_map[opt] for opt in options]
        n_options = len(self.fm.feature_map)

        self.side_constraints.append(
            z3.Sum(
                [
                    z3.ZeroExt(
                        n_options + 1, z3.Extract(opt_id, opt_id, self.fm.target)
                    )
                    for opt_id in option_ids
                ]
            )
            == len(options) - n
        )

    def _solutions_to_dataframe(self, solutions):
        solutions = np.vstack(
            [int_to_config(s.as_long(), len(self.fm.index_map)) for s in solutions]
        )[:, 1:]
        features = [self.fm.index_map[i] for i in self.fm.index_map]
        sample = pd.DataFrame(solutions, columns=features)

        return sample


class RandomSampler(SingleSampler):
    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def promote_diversity(self):
        pass  # stub for executing different techniques


class DFSSampler(RandomSampler):
    """
    This class implements a basic sampling strategy ('naive random') that
    repeatedly searches for solutions to the feature model. Here, we run into
    the problem of 'clustered' solutions, which show variation only among few
    configuration options. This strategy is NOT suited for experiments, but
    to show how SMT solvers can be used for sampling and what problems come
    with doing so.

    The clustering strategies 'DiversityPromotionSampler' and
    'DistanceBasedSampler' aim at mitigating the clustered solution problem.
    The sampling strategy 'BDDSampler' avoids using a SMT solver altogether, but
    does not scale well to large configuration spaces.
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def sample(self, **kwargs):

        if "size" in kwargs:
            n = kwargs["size"]
        else:
            raise AttributeError("Missing argument 'size'.")

        solutions = []
        for i in range(n):
            self.promote_diversity()

            solver = z3.Solver()
            solver.add(self.fm.bitvec_constraints)

            # add side constraints
            solver.add(self.side_constraints)

            for solution in solutions:
                solver.add(self.fm.target != solution)

            if solver.check() == z3.sat:
                solution = solver.model()[self.fm.target]
                solutions.append(solution)

            else:
                print(
                    "Cannot find more than {} valid configurations".format(
                        len(solutions)
                    )
                )
                break

        sample = self._solutions_to_dataframe(solutions)
        return sample


class DiversityPromotionSampler(DFSSampler):
    """
    This class implements uniform random sampling with distance constraints.
    The main idea is that for all configurations, a random number of features
    is selected in order to obtain a representative configuration sample.
    This aims at avoiding bias introduced by the constraint solver.

    Reference:
    C. Kaltenecker, A. Grebhahn, N. Siegmund, J. Guo and S. Apel,
    "Distance-Based Sampling of Software Configuration Spaces,"
    2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE),
    Montreal, QC, Canada, 2019, pp. 1084-1094, doi: 10.1109/ICSE.2019.00112.
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def promote_diversity(self):
        self.fm.shuffle()


class DistanceBasedSampler(SingleSampler):
    """
    This class implements uniform random sampling with distance constraints.
    The main idea is that for all configurations, a random number of features
    is selected in order to obtain a representative configuration sample.
    This aims at avoiding bias introduced by the constraint solver.

    Note: the number of configuration options enabled follows a uniform
    distribution for this implementation!

    Reference:
    C. Kaltenecker, A. Grebhahn, N. Siegmund, J. Guo and S. Apel,
    "Distance-Based Sampling of Software Configuration Spaces,"
    2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE),
    Montreal, QC, Canada, 2019, pp. 1084-1094, doi: 10.1109/ICSE.2019.00112.
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def sample(self, **kwargs):

        if "size" in kwargs:
            size = kwargs["size"]
        else:
            raise AttributeError("Missing argument 'size'.")

        n_options = len(self.fm.index_map)

        # add one option since bitvector is longer than n_opt
        origin = z3.BitVecVal("0" * (n_options + 1), n_options)

        clauses = self.fm.bitvec_constraints
        target = self.fm.target

        clauses = z3.And(clauses)

        # for efficiency purposes, we use one solver instance
        # for each possible distance from the origin
        solvers = {i: z3.Solver() for i in range(1, n_options)}
        for index in solvers.keys():
            solvers[index].add(clauses)

            # add side constraints
            solvers[index].add(self.side_constraints)

            # Distance constraint is expressed as the sum of enabled features
            solvers[index].add(
                z3.Sum(
                    [
                        z3.ZeroExt(n_options + 1, z3.Extract(i, i, target))
                        for i in range(n_options)
                    ]
                )
                == index
            )

        # set of existing solutions
        solutions = []

        # unsatisfiable distances
        available_distances = list(range(1, n_options))
        unsatisfiable = False
        sample_sized_reached = False

        while not (unsatisfiable or sample_sized_reached):
            distance = np.random.choice(available_distances)

            if solvers[distance].check() == z3.sat:
                solution = solvers[distance].model()[target]
                solvers[distance].add(target != solution)
                solutions.append(solution)
            else:
                available_distances.remove(distance)

            unsatisfiable = len(available_distances) == 0
            sample_sized_reached = len(solutions) == size

        sample = self._solutions_to_dataframe(solutions)
        return sample


class CoverageSampler(SingleSampler):
    """
    This class implements sampling strategies with regard to main effects, such as
    single features' or interactions' influences. This comprises both feature-wise, t-wise
    and negative-feature-wise / negative-t-wise sampling.
    Brief summary of sampling strategies:
    Feature-wise sampling:
    Activates one feature at a time while minimizing the total
    number of set of activated features. This usually only activates the mandatory features
    plus those implied by the one feature selected. The total number of configurations should be
    equal to the number of optional features.

    T-wise sampling:
    Similarly to feature-wise sampling, this activates a combination of T features
    per configuration test for this interaction's influence. Again, the total number of selected features
    is minimized, such that only implied and mandatory features are selected. The number of configurations cannot
    be greater than 'N over T', where N is the total number of features and T is the target interaction degree.

    Negative feature-wise sampling:
    Similar to feature-wise sampling with the exception that each optional feature
    is de-selected while the the total number of selected features is maximized. Here, we aim at identifying the
    influence caused by the absence of individual features.

    Negative t-wise sampling:
    Similar to t-wise sampling, but the combination of T features is de-selected and the total
    number of features is maximized. Here, we aim at identifying the influence caused by the absence
    of a feature combination.

    References:
    -
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def sample(self, **kwargs):

        # TODO include minimal configuration

        if "t" not in kwargs:
            raise AttributeError("Missing argument 't'.")
        t = kwargs["t"]

        negwise = False if "negwise" not in kwargs else kwargs["negwise"]

        optionals = self.fm.find_optional_options()["optional"]

        n_options = len(self.fm.index_map)
        target = self.fm.target
        constraints = []
        solutions = []
        for interaction in itertools.combinations(optionals, t):

            # initialize a new optimizer
            optimizer = z3.Optimize()

            # add feature model clauses
            optimizer.add(self.fm.bitvec_constraints)

            # add side constraints
            optimizer.add(self.side_constraints)

            # add previous solutions as constraints
            for solution in solutions:
                optimizer.add(solution != target)

            for opt in interaction:
                if not negwise:
                    constraint = z3.Extract(opt, opt, target) == 1
                else:
                    constraint = z3.Extract(opt, opt, target) == 0

                optimizer.add(constraint)

            # function that counts the number of enabled features
            func = z3.Sum(
                [
                    z3.ZeroExt(n_options, z3.Extract(i, i, target))
                    for i in range(n_options)
                ]
            )

            if not negwise:
                optimizer.minimize(func)
            else:
                optimizer.maximize(func)

            if optimizer.check() == z3.sat:
                solution = optimizer.model()[target]
                constraints.append(solution != target)
                solutions.append(solution)

        sample = self._solutions_to_dataframe(solutions)
        return sample


class BDDSampler(SingleSampler):
    """
    This class implements consistent uniform random sampling by partitioning the configuration space. The idea
    is to construct a binary decision diagram (BDD) for an existing feature model. Each distinct path in the BDD
    represents a partition of the confiuration space. For each path, for some, but not all configuration options values
    are assigned, leaving the un-assigned free to select from. By sampling across such partitions, one obtains a true
    random set of valid configurations.

    The construction of a BBD is time-consuming and does not scale well for large and complex feature models, however,
    this approach asserts randomness of the set of sampled configurations.

    References:
    Jeho Oh, Don Batory, Margaret Myers, and Norbert Siegmund. 2017. Finding near-optimal configurations
    in product lines by random sampling. In Proceedings of the 2017 11th Joint Meeting on Foundations of
    Software Engineering (ESEC/FSE 2017). Association for Computing Machinery, New York, NY, USA, 61–71.
    DOI:https://doi.org/10.1145/3106237.3106273
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def sample(self, **kwargs):

        if "size" not in kwargs:
            raise AttributeError("Missing argument 'size'.")
        sample_size = kwargs["size"]

        partitions = self.fm._compute_partitions()
        n_options = len(self.fm.index_map)

        # calculate proportional size for each partition
        all_configs = sum([2 ** (n_options - len(p)) for p in partitions])
        props = [2 ** (n_options - len(p)) / all_configs for p in partitions]

        samples = []
        for i, p in enumerate(partitions):

            # Calculate how many options we can assign randomly and
            # how much each partition contributes to the entire pool
            # of samples
            n_free_options = n_options - len(p)
            p_sample_size = int(sample_size * props[i])

            sample = []
            while len(sample) < p_sample_size:
                candidate_config = np.random.choice([0, 1], size=n_options + 1)

                # Theoretically, all solutions up to this point should be valid, yet we do not consider
                # the side constraints when constructing the BDD. Thus, we need to validate the solutions
                # with a SMT solver .. :/
                solver = z3.Solver()

                # Contrary to sampling, we ONLY check side constraints
                solver.add(self.side_constraints)

                # add configuration candidate
                for opt_id in range(1, len(candidate_config)):
                    solver.add(
                        z3.Extract(opt_id, opt_id, self.fm.target)
                        == int(candidate_config[opt_id])
                    )

                if solver.check() == z3.sat:
                    sample.append(candidate_config)

            sample = np.vstack(sample)

            df = pd.DataFrame()
            for col in self.fm.feature_map:
                p = self.fm.feature_map[col]
                df[col] = sample[:, p]

            samples.append(df)

        out_sample = pd.concat(samples)
        out_sample = out_sample.astype(bool)

        return out_sample


class ElementaryEffectSampler(MultiSampler):
    """
    This sampling strategy enables sampling pairs of configurations with one
    or more configuration options enabled and disabled, respectively. This can
    be useful when conducting sensitivity analysis on a single configuration
    option or group thereof.

    For instance, to test whether a subject system is sensitive to option A,
    we sample 2*N configurations (N pairs) randomly under the condition that both
    configurations only differ in option A (or more options specified.)

    Reference:
    Saltelli, Andrea, et al. Global sensitivity analysis: the primer.
    John Wiley & Sons, 2008. (Chapters 2.6 and 3)
    """

    def __init__(self, fm: modeling.CNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def sample(self, **kwargs):
        if "options" not in kwargs:
            raise AttributeError("Missing argument 'options'.")
        if "size" not in kwargs:
            raise AttributeError("Missing argument 'size'.")
        options = kwargs["options"]
        max_size = kwargs["size"]

        # TODO refactor method signature

        n_options = len(self.fm.index_map)
        target = self.fm.target
        constraints = []

        clauses = self.fm.bitvec_constraints
        target = self.fm.target
        clauses = z3.And(clauses)

        solutions = {"enabled": [], "disabled": []}

        solution_constraints = []

        i = 0
        while len(solutions["enabled"]) < max_size and i < max_size * 10:
            i += 1
            # Generate two identical 'targets'
            ps, constraints = self.fm.to_partition_constraints(2)

            # Common constraints:
            # differing in options
            solver = z3.Solver()

            # Add constraint:
            # Validity constraints
            solver.add(constraints)

            # Add constraint:
            # Solutions should not be duplicates
            solver.add(solution_constraints)

            # Add constraint:
            # options must be different
            for option in options:
                opt_id = self.fm.feature_map[option]
                solver.add(
                    z3.And(
                        z3.Extract(opt_id, opt_id, ps[0]) == 1,
                        z3.Extract(opt_id, opt_id, ps[1]) == 0,
                    )
                )

            # Add constraint:
            # Let all other options be identical
            for option in set(self.fm.feature_map.keys()) - set(options):
                opt_id = self.fm.feature_map[option]

                solver.add(
                    z3.Extract(opt_id, opt_id, ps[0])
                    == z3.Extract(opt_id, opt_id, ps[1])
                )


            # Add constraints:
            # Number of configuration options enabled
            available_distances = list(range(1, n_options))

            dist_1 = np.random.choice(available_distances)
            dist_2 = dist_1 - len(options)

            # dist_1 for ps[0]
            solver.add(
                z3.Sum(
                    [
                        z3.ZeroExt(n_options + 1, z3.Extract(i, i, ps[0]))
                        for i in range(n_options)
                    ]
                )
                == int(dist_1)
            )

            # dist_2 for ps[1]
            solver.add(
                z3.Sum(
                    [
                        z3.ZeroExt(n_options + 1, z3.Extract(i, i, ps[1]))
                        for i in range(n_options)
                    ]
                )
                == int(dist_2)
            )


            if solver.check() == z3.sat:

                enabled_configuration = solver.model()[ps[0]]
                disabled_configuration = solver.model()[ps[1]]

                solutions["enabled"].append(enabled_configuration)
                solutions["disabled"].append(disabled_configuration)

                # avoid duplicates!!!11!!!!elf
                solution_constraints.append(enabled_configuration != ps[0])
                solution_constraints.append(disabled_configuration != ps[0])
                solution_constraints.append(enabled_configuration != ps[1])
                solution_constraints.append(disabled_configuration != ps[1])
            else:
                pass

        solutions["enabled"] = np.vstack(
            [
                int_to_config(s.as_long(), len(self.fm.index_map))
                for s in solutions["enabled"]
            ]
        )[:, 1:]
        solutions["disabled"] = np.vstack(
            [
                int_to_config(s.as_long(), len(self.fm.index_map))
                for s in solutions["disabled"]
            ]
        )[:, 1:]

        features = [self.fm.index_map[i] for i in self.fm.index_map]
        en = pd.DataFrame(solutions["enabled"], columns=features)
        dis = pd.DataFrame(solutions["disabled"], columns=features)

        return en, dis


class OfflineSampler:
    """
    Wrapper class for sampling strategies on already existing data. The intended use case
    are third-party data sets.
    """

    def __init__(self, df: pd.DataFrame):

        df = df.sample(frac=1).reset_index(drop=True)
        self.df = df

    def elementary_effect_sample(
        self,
        options,
        max_size: int = 30,
        safety_check: bool = True,
    ):
        """
        This method implements a ElementaryEffectSampler for off-line data.
        """
        X = self.df

        # Check whether sample is boolean-ish
        if safety_check:
            dtypes = np.unique(X.dtypes)
            uvalues = np.unique(X)
            if len(dtypes) == 1 and dtypes[0] == np.dtype("bool"):
                pass
            elif len(uvalues) == 2 and (0 in uvalues and 1 in uvalues):
                pass
            else:
                raise AttributeError(
                    "Sample contains numeric values, these are not supported yet!"
                )

        # Some caching
        columns = X.columns

        # Drop duplicates (should not be necessary, but still useful)
        X = X.drop_duplicates()

        # Search for duplicates ignoring the options specified
        columns_for_duplicates = set(X.columns) - set(options)
        dupes = X.duplicated(subset=columns_for_duplicates)
        X = X[dupes]

        # The following boolean operations require different representation
        X = X.astype(bool)

        # Filter rows where specified columns are either 0 or 1
        X["all_enabled"] = np.bitwise_and.reduce(X[options], 1)
        X["all_disabled"] = np.bitwise_and.reduce(np.invert(X[options]), 1)

        # Get rows where specified columns are consistently enabled/disabled
        enabled = X[X["all_enabled"]][columns]
        disabled = X[X["all_disabled"]][columns]

        # Find matching pairs to report indexes for
        df = pd.concat([enabled, disabled])
        enabled_idx = []
        disabled_idx = []

        for _, sample in df.groupby(by=list(set(columns) - set(options))):
            if sample.shape[0] == 2 and len(enabled_idx) < max_size:
                enabled_idx.append(sample.index[0])
                disabled_idx.append(sample.index[1])

        # Report the indexes of pairs
        return enabled_idx, disabled_idx


if __name__ == "__main__":
    pass
