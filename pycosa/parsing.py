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

    def __init__(self, ):
        super().__init__()

    def parse(self, path: str) -> None:

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
                index = int(line[1])
                feature_name = line[2]
                
                # record
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
        raise NotImplementedError()

    def parse(self, path: str) -> None:
        raise NotImplementedError()


class SPLCParser(Parser):

    def __init__(self, ):
        super().__init__()

    def _create_mapping(self, dom):
        raise NotImplementedError()

    def parse(self, path: str) -> None:
        raise NotImplementedError()
        

class SPLOTParser(Parser):

    def __init__(self, ):
        raise NotImplementedError()

    def parse(self, path: str) -> None:
        raise NotImplementedError()



if __name__ == "__main__":
    pass