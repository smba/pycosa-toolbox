from typing import Sequence
import itertools
import networkx as nx
import numpy as np
import z3

from deprecated import deprecated

import networkx as nx
from pyeda.inter import expr, expr2bdd


class CNFExpression:
    def __init__(self):
        self.clauses = None
        self.index_map = None
        self.feature_map = None
        self.bitvec_constraints = None
        self.target = None

    def from_dimacs(self, path: str):
        # parse file
        self.__parse_dimacs(path)

        # build bitvector representation
        self.__to_bitvec()

    def __parse_dimacs(self, path: str) -> (Sequence[Sequence[int]], dict):
        """
        :param path:
        :return:
        """

        dimacs = list()
        dimacs.append(list())
        with open(path) as mfile:
            lines = list(mfile)

            # parse names of features from DIMACS comments (lines starting with c)
            feature_lines = list(filter(lambda s: s.startswith("c"), lines))
            index_map = dict(
                map(
                    lambda l: (int(l.split(" ")[1]), l.split(" ")[2].replace("\n", "")),
                    feature_lines,
                )
            )

            index_map = {idx: index_map[idx] for idx in index_map}

            feature_map = {index_map[v]: v for v in index_map}
            # remove comments form dimacs representation
            lines = list(filter(lambda s: not s.startswith("c"), lines))

            for line in lines:
                tokens = line.split()
                if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                    for tok in tokens:
                        lit = int(tok)
                        if lit == 0:
                            dimacs.append(list())
                        else:
                            dimacs[-1].append(lit)
            assert len(dimacs[-1]) == 0
            dimacs.pop()

        self.clauses = dimacs
        self.index_map = index_map
        self.feature_map = feature_map

    def __to_bitvec(self):
        if self.clauses is not None:
            n_features = len(self.index_map)

            bitvec_constraints = []
            target = z3.BitVec("target", n_features + 1)

            # add clauses of variability model
            for clause in self.clauses:
                c = []
                for opt in clause:
                    opt_sign = 1 if opt >= 0 else 0
                    optid = abs(opt)
                    c.append(z3.Extract(optid, optid, target) == opt_sign)

                bitvec_constraints.append(z3.Or(c))

            # add additional option
            constraint = z3.Extract(0, 0, target) == 0
            bitvec_constraints.append(constraint)

            self.bitvec_constraints = bitvec_constraints
            self.target = target

    def _to_expression(self) -> str:
        """
        This method creates a serialized representation of the DIMACS
        constraints/ the feature model. This is mainly used for partitioning
        the CNF and is likely not very performant.

        Returns
        -------
        String representation of the CNF (compatible with pyEDA module)

        """

        # map variables
        dimacs = self.clauses
        literals = self.index_map

        # Wrap everthing in Strings
        literal = lambda l: "~" + literals[-1 * l] if l < 0 else literals[l]  # + str(l)
        clause = lambda c: "(" + " | ".join(list(map(literal, c))) + ")"
        cnf = " & ".join(list(map(clause, dimacs)))

        return cnf

    def _compute_partitions(self):
        """
        This method computes partitions based on a binary decision
        diagram (BDD) of the DIMACS clauses.

        From the BDD (an acyclic directed graph), we track all paths
        leading to valid configurations. The outcome is a set of
        partitions, from which one can sample uniformly.


        Returns
        -------
        partitions : Sequence
            List of partitions (each item contains information of what options
                                need to be selected/deselected).

        """
        # Create local variables
        expression = self._to_expression()
        features = self.feature_map

        # Create graph representation using NetworkX
        expression = expr(expression)
        bdd = expr2bdd(expression)
        dotrep = bdd.to_dot()
        start = dotrep.find("{") + 2
        end = dotrep.find("}") - 1
        entities = dotrep[start:end].split(";")

        nodes = list(filter(lambda x: "shape" in x, entities))
        nnodes = {}
        for node in nodes:
            node = node.strip()
            name = node.split(" ")[0]
            label = (
                node.split(" ")[1][1:-1]
                .split(",")[0]
                .replace("label=", "")
                .replace('"', "")
            )
            nnodes[name] = label

        edges = list(filter(lambda x: "--" in x, entities))

        nedges = {}
        for edge in edges:
            edge = edge.strip()
            fromto = tuple(edge.split(" [")[0].split(" -- "))
            label = int(edge.split(" [")[1][6:7])
            nedges[fromto] = label

        # Populate graph with information from above
        G = nx.DiGraph()

        for node in nnodes:
            G.add_node(node)

        for edge in nedges:
            G.add_edge(edge[0], edge[1])

        # Track paths to valid configurations:
        # The start node is the node from which edges start, but to
        # which no edge leads.
        start_nodes = set([edge[0] for edge in nedges])
        end_nodes = set([edge[1] for edge in nedges])

        start_node = list(start_nodes - end_nodes)[0]
        end_node = list(filter(lambda x: nnodes[x] == "1", nnodes))[0]

        frequencies = {feature: 0 for feature in features}
        partitions = []
        overall_size = 0
        for path in nx.all_simple_edge_paths(G, start_node, end_node):
            partition_size = 2 ** (len(features) - len(path))
            partition = {}
            overall_size += partition_size
            for edge in path:
                partition[nnodes[edge[0]]] = bool(nedges[edge])
            partitions.append(partition)

        return partitions

    def shuffle(self):
        if self.clauses is not None:

            new_clauses = []
            for clause in self.clauses:
                new_clauses.append(np.random.permutation(clause).tolist())
            self.clauses = np.random.permutation(
                np.array(new_clauses, dtype=object)
            ).tolist()
            self.__to_bitvec()

    @deprecated(reason="This is legacy code used for group sampling")
    def find_alternative_options(self, optional_options):
        mutex_graph = nx.Graph()
        for i, j in itertools.combinations(optional_options, 2):
            solver = z3.Solver()
            solver.add(self.bitvec_constraints)

            constraint_i = z3.And(
                [z3.Extract(i, i, self.target) == 1, z3.Extract(j, j, self.target) == 1]
            )
            solver.add(constraint_i)

            if solver.check() == z3.unsat:
                mutex_graph.add_edge(i, j)

        mutex_groups = []
        for clique in nx.find_cliques(mutex_graph):
            mutex_groups.append(clique)

        return mutex_groups

    @deprecated(reason="This is legacy code used for group sampling")
    def find_optional_options(self):

        deselectable = []
        mandatory = []
        for index, feature_name in self.index_map.items():

            solver = z3.Solver()
            solver.add(self.bitvec_constraints)
            solver.add(z3.Extract(index, index, self.target) == 0)
            if solver.check() == z3.sat:
                deselectable.append(index)
            else:
                mandatory.append(index)

        optionals = []
        dead = []
        for index in deselectable:
            solver = z3.Solver()
            solver.add(self.bitvec_constraints)
            solver.add(z3.Extract(index, index, self.target) == 1)
            if solver.check() == z3.sat:
                optionals.append(index)
            else:
                dead.append(index)

        return {"optional": optionals, "mandatory": mandatory, "dead": dead}

    def to_partition_constraints(self, nps=2):
        """
        This method is used for generating partioning constraints when
        using group-wise sampling.
        Parameters
        ----------
        nps : TYPE, optional
            DESCRIPTION. The default is 2.
        Returns
        -------
        ps : TYPE
            DESCRIPTION.
        bitvec_constraints : TYPE
            DESCRIPTION.
        """
        if self.clauses is not None:
            n_features = len(self.index_map)

            bitvec_constraints = []
            ps = [
                z3.BitVec("partition_{}".format(i), n_features + 1) for i in range(nps)
            ]

            for p in ps:
                # add clauses of variability model
                for clause in self.clauses:
                    c = []
                    for opt in clause:
                        opt_sign = 1 if opt >= 0 else 0
                        optid = abs(opt)
                        c.append(z3.Extract(optid, optid, p) == opt_sign)

                    bitvec_constraints.append(z3.Or(c))

                # add additional option
                constraint = z3.Extract(0, 0, p) == 0
                bitvec_constraints.append(constraint)

            return ps, bitvec_constraints
