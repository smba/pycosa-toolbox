import unittest

import pycosa.parsing as parsing
import pycosa.modeling as modeling
import pycosa.sampling as sampling
import pycosa.util as util
import pycosa.lib as lib

import itertools

import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt


class TestSampler(unittest.TestCase):
    def setUp(self):
        pass

    def testFoo(self):
        parser = parsing.DimacsParser()
        parser.parse("_test_data/feature_models/h2.dimacs")

        vm = modeling.VariabilityModel()
        vm.from_parser(parser)

        sampler = sampling.GroupSampler(vm)
        en, dis = sampler.sample(["RECOMPILE_ALWAYS", "COMPRESS"], size=20)
        options = vm.get_binary_features()

        plt.pcolormesh(en)
        plt.show()

if __name__ == "__main__":
    parser = parsing.DimacsParser()
    parser.parse("_test_data/feature_models/h2.dimacs")

    vm = modeling.VariabilityModel()
    vm.from_parser(parser)

    sampler = sampling.GroupSampler(vm)
    en, dis = sampler.sample(options =["COMPRESS", "DROP_RESTRICT"], max_size=100)
    plt.pcolormesh(en)

        