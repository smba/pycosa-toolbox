"""
Identification of influential and interacting options from group-wise
elementary-effect measurements.

:class:`GroupLearner` tracks for each option how often measurements of
groups containing it indicate an influential or an interacting option, and
proposes new groups to measure next.
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from pycosa.lib import OptionExhaustionError


class GroupLearner:
    """
    Classifies options as influential and/or interacting based on
    group-wise elementary-effect measurements.

    Options are always measured in groups: for a group, the performance of
    configurations with all group options enabled is compared against the
    performance of pair-wise configurations with all group options
    disabled (see :class:`pycosa.sampling.GroupSampler`). Whether a group
    counts as influential or interacting follows the thresholds ``t1``
    (relative performance difference) and ``t2`` (relative spread of the
    differences), akin to elementary effects / Morris screening.
    """

    def __init__(self, options, t1: float = 0.5, t2: float = 0.1, seed=None):
        """
        Keeps track of which options are classified as influential and/or
        interacting, based on group-wise elementary effect measurements.

        Parameters
        ----------
        options : iterable of str
            Names of the options under investigation.
        t1 : float, optional
            Relative threshold for a performance difference to count as
            relevant (default 0.5).
        t2 : float, optional
            Relative threshold for the spread of performance differences
            (default 0.1).
        seed : int, optional
            Seed for reproducible group suggestions.
        """
        self.rng = np.random.default_rng(seed)
        self.t1 = t1
        self.t2 = t2

        self.options = set(options)

        self.records = {
            option: {"influential": 0, "interacting": 0} for option in self.options
        }

    def classify(
        self, group: Sequence[str], perf_en: np.array, perf_dis: np.array
    ) -> None:
        """
        Decides for a given set of options and measurements, whether the set of options
        is influential and/or interacting with further features.

        Parameters
        ----------
        group : sequence of str
            Options that were measured together.
        perf_en : array-like
            Performance of configurations with the group options enabled.
        perf_dis : array-like
            Performance of the matching configurations with the group
            options disabled; must have the same length as ``perf_en``.

        Returns
        -------
        None
        """

        perf_en = np.asarray(perf_en, dtype=float)
        perf_dis = np.asarray(perf_dis, dtype=float)
        if perf_en.shape != perf_dis.shape or perf_en.size == 0:
            raise ValueError(
                "perf_en and perf_dis must be non-empty and of equal length"
            )

        n_pairs = len(perf_en)
        deltas = np.abs(perf_en - perf_dis)
        mean_delta = np.mean(deltas)

        # Collect pairs where the performance difference is less than or equal to a percentage (t1)
        # of the observed performance.
        less_or_equals_t1 = [
            deltas[i] <= self.t1 * min(abs(perf_en[i]), abs(perf_dis[i]))
            for i in range(n_pairs)
        ]

        # Collect pairs where the performance difference is greater than to a percentage (t1)
        # of the observed performance.
        greater_than_t1 = [(not g) for g in less_or_equals_t1]

        # Collect pairs where the performance difference is greater than to a percentage (t2)
        # of the observed mean performance difference.
        greater_than_t2 = [
            abs(delta - mean_delta) > (self.t2 * mean_delta) for delta in deltas
        ]
        t2_check = np.mean(deltas) > self.t2

        # We distinguish four different cases
        if all(greater_than_t1):
            # Case 1: All effects appear to be non-zero (hence, influential), but some
            # effects show wide spread (hence, some are interacting)
            # if any(greater_than_t2):
            #    self.__record_influentials(group)
            #    self.__record_interactings(group)

            if t2_check:
                self.__record_interactings(group)

            # Case 2: All effects appear to be non-zero (hence, influential), none
            # is interacting
            else:
                self.__record_influentials(group)

        # Case 3: Some effects observed are relevant, but not all, hence some options
        # are interacting while others are non-influential.
        elif any(less_or_equals_t1) and any(greater_than_t1):
            self.__record_interactings(group)

        # Case 4: The observed effects are all neglible, hence, no option
        # can be considered influential/interacting
        elif all(less_or_equals_t1):
            self.__drop_options(group)

    def __record_influentials(self, options: Sequence[str]) -> None:
        """
        Keep track of options that are classified as influential.

        Parameters
        ----------
        options : sequence of str
            Options to keep track of.
        """
        for option in options:
            self.records[option]["influential"] += 1

    def __record_interactings(self, options: Sequence[str]) -> None:
        """
        Keep track of options that are classified as interacting.

        Parameters
        ----------
        options : sequence of str
            Options to keep track of.
        """
        for option in options:
            self.records[option]["interacting"] += 1

    def __drop_options(self, options: Sequence[str]) -> None:
        """
        Drops the given options, which appear non-influential, from the
        pool of options to be regrouped.

        Parameters
        ----------
        options : sequence of str
            Options to drop.
        """
        self.options = self.options - set(options)

    def suggest_options(self, size: int = 5) -> np.array:
        """
        Suggests a number of options to be included in a new group.

        Parameters
        ----------
        size : int, optional
            Number of options to include in the group (default 5).

        Returns
        -------
        numpy.ndarray
            The suggested options. If no options remain, an
            :class:`pycosa.lib.OptionExhaustionError` is raised.
        """

        # Either pick all or a specified number of options for group
        group_size = min(len(self.options), size)

        if group_size > 0:
            options = self.rng.choice(
                sorted(self.options), size=group_size, replace=False
            )
        else:
            msg = "Not enough options left to suggest a group (previous ones have already been discarded.)"
            raise OptionExhaustionError(msg)

        return options

    def get_statistics(self) -> pd.DataFrame:
        """
        Return the current statistics of which options are considered influential and/or interacting.

        Returns
        -------
        pd.DataFrame
            Counts of influential/interacting classifications per option.
        """
        df = pd.DataFrame.from_dict(self.records, orient="index")
        df.index.name = "option"

        return df


if __name__ == "__main__":
    pass
