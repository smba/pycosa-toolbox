#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parsers that translate variability models from various formats into
DIMACS-style clauses with named options (see
:class:`pycosa.modeling.VariabilityModel`).
"""

import itertools
import logging
import pathlib
import re
from typing import Dict, List, Sequence

import xmlschema


class Parser:
    """
    Base class for feature-model parsers.

    A parser translates a variability model into DIMACS-style clauses and
    establishes a bidirectional mapping between variable indexes and
    option names. The result is consumed by
    :meth:`pycosa.modeling.VariabilityModel.from_dimacs`.
    """

    def __init__(self):
        """Creates an empty parser; call :meth:`parse` to fill it."""
        self._index_to_feature: Dict[int, str] = dict()
        self._feature_to_index: Dict[str, int] = dict()
        self._clauses: List[List[int]] = []

    def _check_parsed(self):
        """Raises if :meth:`parse` has not been called successfully yet."""
        if len(self._index_to_feature) == 0 and len(self._clauses) == 0:
            msg = "No feature model parsed yet! Use {}.parse() first."
            raise RuntimeError(msg.format(self.__class__.__name__))

    def _register(self, index: int, feature_name: str) -> None:
        """Adds a variable index/option name pair to both mappings."""
        self._index_to_feature[index] = feature_name
        self._feature_to_index[feature_name] = index

    def get_feature(self, index: int) -> str:
        """Returns the name of the option with the given variable index."""
        self._check_parsed()
        if index in self._index_to_feature:
            return self._index_to_feature[index]
        raise ValueError("No feature with index '{}' found!".format(index))

    def get_index(self, feature_name: str) -> int:
        """Returns the variable index of the given option."""
        self._check_parsed()
        if feature_name in self._feature_to_index:
            return self._feature_to_index[feature_name]
        msg = "No feature index for feature name '{}' found!".format(feature_name)
        raise ValueError(msg)

    def get_clauses(self) -> List[List[int]]:
        """Returns the parsed clauses in DIMACS style."""
        return self._clauses

    def get_features(self) -> List[str]:
        """Returns the names of all parsed options."""
        return list(self._feature_to_index.keys())

    def get_features2index(self) -> Dict[str, int]:
        """Returns the mapping from option names to variable indexes."""
        return self._feature_to_index

    def get_index2features(self) -> Dict[int, str]:
        """Returns the mapping from variable indexes to option names."""
        return self._index_to_feature

    def parse(self, path: str) -> None:
        """Parses the file at ``path``; implemented by subclasses."""
        raise NotImplementedError()


class DimacsParser(Parser):
    """
    Parser for DIMACS CNF files. Feature names are read from comment lines of
    the form ``c <index> <name>`` (as written by FeatureIDE, KConfigReader,
    etc.). FeatureIDE marks auxiliary variables with a trailing ``$`` after the
    index, which is accepted as well. Variables without a name are named
    ``_x<index>``.
    """

    _FEATURE_COMMENT = re.compile(r"^c\s+(\d+)\$?\s+(\S.*?)\s*$")

    def __init__(self):
        """Creates an empty parser; call :meth:`parse` to fill it."""
        super().__init__()
        self.n_variables = None
        self.n_clauses = None

    def parse(self, path: str) -> None:
        """Parses the DIMACS file at ``path`` (replacing previous results)."""
        with open(path, "r") as file:
            self.parse_string(file.read())

    def parse_string(self, text: str) -> None:
        """Parses DIMACS CNF text (replacing previous results).

        Clauses may span multiple lines, and a line may contain several
        clauses, each terminated by the literal 0. A clause that is not
        terminated by 0 is accepted if it ends at the end of the input.
        Variables without a name in the comments are named ``_x<index>``.
        """
        self._index_to_feature, self._feature_to_index = dict(), dict()
        self._clauses = []

        current: List[int] = []
        for raw in text.splitlines():
            line = raw.strip()
            if line == "" or line.startswith("%"):
                continue

            if line.startswith("c"):
                match = self._FEATURE_COMMENT.match(line)
                if match is not None:
                    self._register(int(match.group(1)), match.group(2))
                continue

            if line.startswith("p"):
                tokens = line.split()
                if len(tokens) >= 4:
                    self.n_variables = int(tokens[2])
                    self.n_clauses = int(tokens[3])
                continue

            # clause line; a clause may span multiple lines and a line may
            # contain multiple clauses, each terminated by 0
            for token in line.split():
                literal = int(token)
                if literal == 0:
                    self._clauses.append(current)
                    current = []
                else:
                    current.append(literal)

        if len(current) > 0:
            # tolerate a missing trailing 0
            self._clauses.append(current)

        # name unnamed variables
        indexes = {abs(lit) for clause in self._clauses for lit in clause}
        if self.n_variables is not None:
            indexes |= set(range(1, self.n_variables + 1))
        for index in sorted(indexes):
            if index not in self._index_to_feature:
                self._register(index, "_x{}".format(index))

        if self.n_clauses is not None and self.n_clauses != len(self._clauses):
            logging.warning(
                "DIMACS header declares %d clauses, but %d were found.",
                self.n_clauses,
                len(self._clauses),
            )


class SPLCParser(Parser):
    """
    Parser for SPL Conqueror variability models (XML). Only binary options are
    translated into clauses; numeric options are recorded in
    ``self.numeric_options``. Boolean constraints are supported if they are
    conjunctions of clauses written with ``|`` (or), ``&`` (and), ``!``/``-``
    (not), and ``=>`` (implication between two literals).
    """

    def __init__(self):
        """Creates an empty parser; call :meth:`parse` to fill it."""
        super().__init__()
        here = pathlib.Path(__file__).parent.resolve()
        self.schema = xmlschema.XMLSchema(str(here / "resources" / "splc.xsd"))
        self.numeric_options: Dict[str, dict] = {}

    @staticmethod
    def _as_list(value) -> list:
        """Normalizes XML children that may occur once or several times
        into a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def _option_list(cls, node) -> List[str]:
        """Extracts the whitespace-trimmed option names from an XML node."""
        if not isinstance(node, dict):
            return []
        return [o.strip() for o in cls._as_list(node.get("option")) if o and o.strip()]

    def _parse_constraint(self, constraint: str) -> List[List[int]]:
        """Translates one boolean constraint of the XML into clauses.

        The constraint must be a conjunction (``&``) of clauses written with
        ``|`` (or), ``!``/``-`` (not), and ``=>`` (implication). Unknown
        option names raise a ValueError.
        """
        clauses = []
        for conjunct in constraint.split("&"):
            conjunct = conjunct.strip().strip("()").strip()
            if conjunct == "":
                continue
            if "=>" in conjunct:
                lhs, rhs = conjunct.split("=>", 1)
                conjunct = "!{} | {}".format(lhs.strip(), rhs.strip())

            clause = []
            for literal in conjunct.split("|"):
                literal = literal.strip().strip("()").strip()
                negated = False
                while literal.startswith(("!", "-")):
                    negated = not negated
                    literal = literal[1:].strip()
                if literal not in self._feature_to_index:
                    raise ValueError(
                        "Unknown option '{}' in constraint '{}'".format(
                            literal, constraint
                        )
                    )
                index = self._feature_to_index[literal]
                clause.append(-index if negated else index)
            clauses.append(clause)
        return clauses

    def parse(self, path: str) -> None:
        """Parses the SPL Conqueror XML file at ``path``.

        Refines parent/child relations, implied and excluded options, and
        boolean constraints into clauses; numeric options are recorded in
        ``self.numeric_options``.
        """
        xml = self.schema.to_dict(path)

        self._index_to_feature, self._feature_to_index = dict(), dict()
        self._clauses = []
        self.numeric_options = {}

        options = self._as_list(
            (xml.get("binaryOptions") or {}).get("configurationOption")
        )
        for index, option in enumerate(options):
            self._register(index + 1, option["name"].strip())

        idx = self._feature_to_index
        seen_groups = set()
        for option in options:
            name = option["name"].strip()
            parent = (option.get("parent") or "").strip()
            optional = str(option.get("optional", "False")).strip().lower() == "true"

            excluded = self._option_list(option.get("excludedOptions"))

            if parent != "":
                # child implies parent
                self._clauses.append([-idx[name], idx[parent]])
                if not optional and not excluded:
                    # mandatory child: parent implies child (non-optional
                    # options with exclusions form alternative groups instead)
                    self._clauses.append([-idx[parent], idx[name]])
            elif not optional:
                # mandatory top-level option (e.g. root)
                self._clauses.append([idx[name]])

            for implied in self._option_list(option.get("impliedOptions")):
                self._clauses.append([-idx[name], idx[implied]])

            if excluded:
                group = tuple(sorted(set(excluded + [name])))
                if group not in seen_groups:
                    seen_groups.add(group)
                    for a, b in itertools.combinations(group, 2):
                        self._clauses.append([-idx[a], -idx[b]])
                    # in SPL Conqueror, a non-optional option that excludes its
                    # siblings forms an alternative group: one must be chosen
                    if not optional and parent != "":
                        self._clauses.append([-idx[parent]] + [idx[o] for o in group])

        for option in self._as_list(
            (xml.get("numericOptions") or {}).get("configurationOption")
        ):
            self.numeric_options[option["name"].strip()] = option

        constraints = (xml.get("booleanConstraints") or {}).get("constraint")
        for constraint in self._as_list(constraints):
            if constraint and constraint.strip():
                self._clauses.extend(self._parse_constraint(constraint))


if __name__ == "__main__":
    pass
