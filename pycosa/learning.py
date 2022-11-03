import logging
from typing import Sequence

import numpy as np
import pandas as pd

from pycosa.lib import OptionExhaustionError


class GroupLearner:
    def __init__(self, options, t1: float = 0.5, t2: float = 0.1):
        """
        Decides for a given set of options and measurements, whether the set of options
        is influential and/or interacting with further features.

        @param group: List of option names
        @param perf_en: Performance of configurations with options of group enabled
        @param perf_dis: Performance of configurations with options of group disabled

        @return: None
        """
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

        @param group: List of option names
        @param perf_en: Performance of configurations with options of group enabled
        @param perf_dis: Performance of configurations with options of group disabled

        @return: None
        """

        n_pairs = len(perf_en)
        deltas = np.abs(perf_en - perf_dis)
        mean_delta = np.mean(deltas)

        # Collect pairs where the performance difference is less than or equal to a percentage (t1)
        # of the observed performance.
        # TODO remove 3 and replace with meaningful threshold
        less_or_equals_t1 = [
            deltas[i] <= self.t1 * min(abs(perf_en[i]), abs(perf_dis[i]))
            or deltas[i] < 3
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

        # We distinguish four different cases
        if all(greater_than_t1):

            # Case 1: All effects appear to be non-zero (hence, influential), but some
            # effects show wide spread (hence, some are interacting)
            if any(greater_than_t2):
                self.__record_influentials(group)
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

        @param List of options to keep track of.
        """
        for option in options:
            self.records[option]["influential"] += 1

    def __record_interactings(self, options: Sequence[str]) -> None:
        """
        Keep track of options that are classified as interacting.

        @param List of options to keep track of.
        """
        for option in options:
            self.records[option]["interacting"] += 1

    def __drop_options(self, options: Sequence[str]) -> None:
        """
        Drop passed options from pool for re-grouping.

        @param List of options to drop
        """
        self.options = self.options - set(options)

    def suggest_options(self, size: int = 5) -> np.array:
        """
        Suggests a number of options to be included in a new group.

        @param size: Number of options to select for group (default is 5)
        @return: Options to include in
        """

        # Either pick all or a specified number of options for group
        group_size = min(len(self.options), size)

        if group_size > 0:
            options = np.random.choice(
                list(self.options), size=group_size, replace=False
            )
        else:
            msg = f"Not enough options left to suggest a group (previous ones have already been discarded.)"
            raise OptionExhaustionError(msg)

        return options

    def get_statistics(self) -> pd.DataFrame:
        """
        Return the current statistics of which options are considered influential and/or interacting.

        Returns
        -------
        @return: pd.DataFrame with statistics
        """
        df = pd.DataFrame(self.records, columns=self.records.keys(), index=False)

        return df


if __name__ == "__main__":
    pass
