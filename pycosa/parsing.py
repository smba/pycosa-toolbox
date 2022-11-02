#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence
import xmlschema
import itertools
import z3

class Parser:
    def __init__(self):

        self._index_to_feature = dict()
        self._feature_to_index = dict()
        #self.clauses = []

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

        self._index_to_feature = dict()
        self._feature_to_index = dict()
        self._clauses = []
xmlschema
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

            elif start == "p":  # p = .. something, used for validation

                line = line.split(" ")
                n_options = int(line[2])
                n_clauses = int(line[3])

                assert len(self._index_to_feature) == len(self._feature_to_index)
                assert len(self._index_to_feature) == n_options

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
        self.schema = xmlschema.XMLSchema("../_test_data/meta/splc.xsd")
        
    def create_alternative_group(self, mutex_options):
        constraints = [] 
        for option in mutex_options:
            options_ = list(set(mutex_options) - set([option]))
            constraints.append(
                 z3.And([z3.Bool(option)] + [z3.Not(z3.Bool(opt)) for opt in options_])   
            )
        return z3.Or(constraints)
    
    def dimacs_create_alternative_group(self, mutex_options):
        constraints = []
        for a, b in itertools.combinations(mutex_options, 2):
            constraints.append(
                [-1*a, -1*b]
            )
        return constraints

    def create_or_group(self, options):
        constraints = z3.Or(
            [z3.Bool(option) for option in options]
        )
        return z3.Or(constraints)
    
    def dimacs_create_or_group(self, options):
        constraints = [option for option in options]
        return constraints
    
    def create_parent_constraint(self, child, parent):
        return z3.Implies(z3.Bool(child), z3.Bool(parent))
    
    def dimacs_create_parent_constraint(self, child, parent):
        return [-1*child, parent]

    def create_mandatory_constraint(self, option):
        return z3.Bool(option)   
    
    def dimacs_create_mandatory_constraint(self, option):
        print(option)
        return [option] 
    
    def parse(self, path: str) -> None:
        
        xml = self.schema.to_dict(path)
        
        constraints = []
        literals = []
        
        dimacs = []
        
        self.options = {}
        
        mutex_groups = []
        for index, option in enumerate(xml['binaryOptions']['configurationOption']):
            name = option["name"]
            self.options[name] = index + 1
        
        for option in xml['binaryOptions']['configurationOption']:
            name = option["name"]
            parent = option["parent"]
            literals.append(z3.Bool(name))
            
            if parent.strip() != "":
                #print("parent", parent)
                # ADD parent constraint
                constraints.append(
                    self.create_parent_constraint(name, parent)
                )
                dimacs.append(self.dimacs_create_parent_constraint(
                    self.options[name], 
                    self.options[parent]
                ))
            
            # check if option is optional
            is_optional = option['optional']
            if not is_optional:
                constraints.append(
                    self.create_mandatory_constraint(name)    
                )
                dimacs.append(
                    self.options[name]
                )

            if option['excludedOptions'] is not None:
                mutexes = option['excludedOptions']['options']
                constraints.append(
                    self.create_alternative_group(mutexes)
                )
                dimacs.append(self.dimacs_create_alternative_group(
                    [self.options[i] for i in mutexes]    
                ))

        return literals, constraints, dimacs
    
if __name__ == "__main__":
    pass