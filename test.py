import unittest

import pycosa.parsing as parsing
import pycosa.modeling as modeling
import pycosa.learning as learning
import pycosa.sampling as sampling
from pycosa.experiments import AttributedVariabilityModelGenerator
import pycosa.util as util
import pycosa.lib as lib

import itertools

import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = parsing.DimacsParser()
    parser.parse("_test_data/feature_models/unconstrained.dimacs")

    vm = modeling.VariabilityModel()
    vm.from_parser(parser)

    avm = AttributedVariabilityModelGenerator(vm)
    d = learning.GroupLearner(vm.get_binary_features())
    for i in range(100):
        print(i)
        sampler = sampling.GroupSampler(vm)
        options = np.random.choice(vm.get_binary_features(), size=3)
        #print(options)
        #en, dis = sampler.sample(options, 50)
        configs = np.random.choice([0,1], size=(30, len(vm.get_binary_features())))
        en, dis = np.copy(configs), np.copy(configs)
        
        en = pd.DataFrame(en, columns=vm.get_binary_features())
        dis = pd.DataFrame(dis, columns=vm.get_binary_features())

        for option in     options:
            en.loc[:,option] = 1
            dis.loc[:,option] = 0
        
        perf1 = avm.get_performances(en.values)
        perf2 = avm.get_performances(dis.values)
        
        delta = np.abs(perf1 - perf2)
        
        d._classify(options, perf1, perf2)

    print(avm.performance_model.terms)
    fig, axes = plt.subplots(2, 1, sharex=True)
    effects = pd.DataFrame(d.records)
    axes[0].bar(np.arange(len(vm.get_binary_features())), effects.iloc[0,:])
    axes[1].bar(np.arange(len(vm.get_binary_features())), effects.iloc[1,:])

    plt.show()
        