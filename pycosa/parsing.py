#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence
import xmltodict
import z3

class Parser:
    def __init__(self):
        
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

    def get_features(self) -> Sequence[str]:
        
        return list(self._feature_to_index.keys())

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
        self._feature_to_index = None
        self._index_to_feature = None

    def _create_mapping(self, dom):
        
        self._feature_to_index = dict()
        self._index_to_feature = dict()

        binary_options = dom['vm']['binaryOptions']['configurationOption']

        for index, opt in enumerate(binary_options):

            # since DIMACS starts from 1
            index += 1 

            # record index and feature names
            self._feature_to_index[ opt['name'] ] = index
            self._index_to_feature[ index ] = opt['name']

    def parse(self, path: str) -> None:
        
        with open(path, 'r') as file:
            file_content = file.read()
            dom = xmltodict.parse(file_content)

        # create mapping of feature names and indexes
        self._create_mapping(dom)

        binary_options = dom['vm']['binaryOptions']['configurationOption']
        for opt in binary_options:

            name = opt['name']
            is_optional = opt['optional']

            # 1) check for parent features --> add an implication THIS --> PARENT
            parent = None
            if opt['parent'] is not None:
                parent = opt['parent']

            # 2) check for excluded features
            excluded = []
            if opt['excludedOptions'] is not None:
                excluded = opt['excludedOptions']['options']
                if type(excluded) is str:
                    excluded = [excluded]

            # 3) check for implied options
            implied = []
            if opt['impliedOptions'] is not None:
                implied = opt['impliedOptions']['options']
                if type(implied) is str:
                    implied = [implied]

            # 4) chec if feature has children
            children = []
            if opt['children'] is not None:
                children = opt['children']['options']
                if type(children) is str:
                    children = [children]
        

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


if __name__ == "__main__":
    parser = SPLConqParser()
    parser.parse('FeatureModel.xml')
    features = parser.get_features()
    parser.get_feature(99)