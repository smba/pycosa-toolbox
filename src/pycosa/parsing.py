#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence

import xmlschema
import itertools
import pathlib


class Parser:
    def __init__(self):

        self._index_to_feature = dict()
        self._feature_to_index = dict()
        self._clauses = []

    def get_feature(self, index: int) -> str:

        if self._index_to_feature is None:
            msg = "No feature model parsed yet! Use {}.parse() first."
            msg = msg.format(self.__class__)
            raise RuntimeError(msg)

        if index in self._index_to_feature:
            return self._index_to_feature[index]
        else:
            msg = "No feature with index '{}' found!".format(index)
            raise ValueError(msg)

    def get_index(self, feature_name: str) -> int:

        if self._feature_to_index is None:
            msg = "No feature model parsed yet! Use {}.parse() first."
            msg = msg.format(self.__class__)
            raise RuntimeError(msg)

        if feature_name in self._feature_to_index:
            return self._feature_to_index[feature_name]
        else:
            msg = "No feature index for feature name '{}' found!".format(feature_name)
            raise ValueError(msg)

    def get_clauses(self) -> Sequence[Sequence[int]]:
        return self._clauses

    def get_features(self) -> Sequence[str]:
        return list(self._feature_to_index.keys())

    def get_features2index(self):
        return self._feature_to_index

    def get_index2features(self):
        return self._index_to_feature

    def parse(self, path: str) -> None:
        raise NotImplementedError()


class DimacsParser(Parser):
    def __init__(
        self,
    ):
        super().__init__()

    def parse(self, path: str) -> None:

        with open(path, "r") as file:
            lines = file.readlines()

        lines = [line.replace("\n", "") for line in lines]

        for line in lines:

            # TODO replace with pattern matching for Python >= 3.10
            start = line[0]

            if start == "c":  # c = comment, used for specifying a feature
                line = line.split(" ")
                index = int(line[1])
                feature_name = line[2]

                # record
                self._index_to_feature[index] = feature_name
                self._feature_to_index[feature_name] = index

            elif start == "p":  # p = .. used for validation only
                pass

            else:  # any other line should specify a clause
                line = line.split(" ")
                assert int(line[-1]) == 0

                clause = [int(literal) for literal in line[:-1]]
                self._clauses.append(clause)


class SPLCParser(Parser):
    def __init__(
        self,
    ):
        super().__init__()
        here = pathlib.Path(__file__).parent.resolve()
        self.schema = xmlschema.XMLSchema(str(here) + "/resources/splc.xsd")

    def _alternative_group(self, mutex_options):
        constraints = []
        for a, b in itertools.combinations(mutex_options, 2):
            constraints.append([-1 * a, -1 * b])
        return constraints

    def _or_group(self, options):
        constraints = [o for o in options]
        return constraints

    def _parent(self, child, parent):
        return [-1 * child, parent]

    def _mandatory(self, option):
        return [option]

    def parse(self, path: str) -> None:

        xml = self.schema.to_dict(path)

        constraints = []
        literals = []

        self._clauses = []

        self.options = {}

        mutex_groups = []
        for index, option in enumerate(xml["binaryOptions"]["configurationOption"]):
            name = option["name"]
            self.options[name] = index + 1

        for option in xml["binaryOptions"]["configurationOption"]:
            name = option["name"]
            parent = option["parent"]
            literals.append(z3.Bool(name))

            if parent.strip() != "":
                self._clauses.append(
                    self.dimacs_create_parent_constraint(
                        self.options[name], self.options[parent]
                    )
                )

            # check if option is optional
            is_optional = option["optional"]
            if not is_optional:
                self._clauses.append(self.options[name])

            if option["excludedOptions"] is not None:
                mutexes = option["excludedOptions"]["options"]

                self._clauses.append(
                    self.dimacs_create_alternative_group(
                        [self.options[i] for i in mutexes]
                    )
                )


if __name__ == "__main__":
    pass
