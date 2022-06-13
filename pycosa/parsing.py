#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence
import z3

class Parser:
    def __init__(self):
        pass

    def get_features(self) -> Sequence[int]:
        pass

    def parse(self, path: str) -> None:
        pass


class CNFExpression():

    def __init__(self):
        """
        Wrapper class for boolean expressions. Since we 

        Returns
        -------
        None.

        """
        pass


class DimacsParser(Parser):

    def __init__(self, ):
        """


        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self._index_to_feature = None
        self._feature_to_index = None
        self.clauses = []

    def get_feature(self, index: int) -> str:
        """


        Parameters
        ----------
        index : int
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        str
            DESCRIPTION.

        """

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
        """


        Parameters
        ----------
        feature_name : str
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        int
            DESCRIPTION.

        """

        if self._feature_to_index is None:
            msg = "No feature model parsed yet! Use {}.parse() first."
            msg = msg.format(self.__class__)
            raise RuntimeError(msg)

        if feature_name in self._feature_to_index:
            return self._feature_to_index[feature_name]
        else:
            msg = "No feature index for feature name '{}' found!".format(feat)
            raise ValueError(msg)

    def get_clauses(self) -> Sequence[Sequence[int]]:
        """


        Returns
        -------
        Sequence[Sequence[int]]
            DESCRIPTION.

        """
        return self._clauses

    def parse(self, path: str) -> None:
        """


        Parameters
        ----------
        path : str
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """

        self._index_to_feature = dict()
        self._feature_to_index = dict()
        self._clauses = []

        with open(path, 'r') as file:
            lines = file.readlines()

        lines = [line.replace('\n', '') for line in lines]

        for line in lines:

            # TODO replace with pattern matching for Python >= 3.10
            start = line[0]

            if start == 'c':  # c = comment, used for specifying a feature
                line = line.split(' ')
                index = line[1]
                feature_name = line[2]

                # populate mappings
                self._index_to_feature[index] = feature_name
                self._feature_to_index[feature_name] = index

            elif start == 'p':  # p = .. something, used for validation
                line = line.split(' ')
                n_options = int(line[2])
                n_clauses = int(line[3])

                assert len(self._index_to_feature) == len(
                    self._feature_to_index)
                assert len(self._index_to_feature) == n_options

            else:  # any other line should specify a clause
                line = line.split(' ')

                assert int(line[-1]) == 0

                clause = [int(literal) for literal in line[:-1]]
                self._clauses.append(clause)


class FeatureIdeParser(Parser):

    def __init__(self, ):
        """
        

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def parse(self, path: str) -> None:
        pass

    def to_cnf(self) -> CNFExpression:
        pass


class SPLConqParser(Parser):

    def __init__(self, ):
        """
        

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def parse(self, path: str) -> None:
        pass

    def to_cnf(self) -> CNFExpression:
        pass


class SplotParser(Parser):

    def __init__(self, ):
        """
        

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def parse(self, path: str) -> None:
        pass

    def to_cnf(self) -> CNFExpression:
        pass


parser = DimacsParser()
parser.parse("model.dimacs")
print(parser.get_clauses())