import itertools
from typing import Sequence

from pycosa.parsing import Parser

import numpy as np
import z3


class VariabilityModel:
    def __init__(self):

        self.binary_options = dict()

        self.dimacs = None
        self.binary_clauses = None
        self.bin_feature_to_index = None
        self.bin_index_to_feature = None

    def from_parser(self, parser: Parser):

        self.bin_feature_to_index = parser.get_features2index()
        self.bin_index_to_feature = parser.get_index2features()

        self.dimacs = parser.get_clauses()
        print(self.dimacs)
        clauses = parser.get_clauses()

        z3_clauses = self._generate_z3_clauses(clauses)

        self.binary_clauses = z3_clauses
        self.binary_options = parser.get_features()

    def _generate_z3_clauses(self, clauses):
        z3_clauses = []
        for clause in clauses:
            ors = []
            for literal in clause:
                sign = literal > 0
                index = abs(literal)
                feature_name = self.bin_index_to_feature[index]

                if sign:
                    add = z3.Bool(feature_name)
                else:
                    add = z3.Not(z3.Bool(feature_name))

                ors.append(add)

            if len(ors) > 1:
                z3_clauses.append(z3.Or(ors))
            else:
                z3_clauses.append(ors[0])

        return z3_clauses

    def get_binary_clauses(self):
        return self.binary_clauses

    def get_binary_features(self):
        return self.binary_options

    def shuffle(self):
        clauses = []
        for clause in self.dimacs:
            idx = np.random.choice(len(clause), replace=False, size=len(clause))
            new_clause = [clause[i] for i in idx]
            clauses.append(new_clause)

        idx = np.random.choice(len(clauses), replace=False, size=len(clauses))
        new_clauses = [clauses[i] for i in idx]

        self.dimacs = new_clauses
        self.binary_clauses = self._generate_z3_clauses(new_clauses)


if __name__ == "__main__":
    pass
