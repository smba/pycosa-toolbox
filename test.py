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
    np.random.seed(1)
    vm = modeling.VariabilityModel()
    vm.from_parser(parser)

    avm = AttributedVariabilityModelGenerator(vm)
    d = learning.GroupLearner(vm.get_binary_features())
    
    n_groups = 150
    for i in range(n_groups):
        #print(i)
        sampler = sampling.GroupSampler(vm)
        
        
        #if i < n_groups // :
        #    options = np.random.choice(vm.get_binary_features(), size=3)
        #else:
        options = d.suggest_options(3)
        #print(options)
        #en, dis = sampler.sample(options, 50)
        configs = np.random.choice([0,1], size=(20, len(vm.get_binary_features())))
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
        
    terms = avm.performance_model.terms
    terms = [item for sublist in terms for item in sublist]
    
    fig, axes = plt.subplots(2, 1, sharex=True)
    effects = pd.DataFrame(d.records)
    
    summ = effects.iloc[0,:]  + effects.iloc[1,:] 
    
    axes[0].bar(np.arange(len(vm.get_binary_features())), effects.iloc[0,:])
    axes[1].bar(np.arange(len(vm.get_binary_features())), effects.iloc[1,:])
    #axes[2].bar(np.arange(len(vm.get_binary_features())), summ, color="brown")
    
    print(terms)
    for t in terms:
        axes[0].axvline(t, color="brown")
        axes[1].axvline(t, color="brown")
        #axes[2].axvline(t, color="lime")
    #plt.bar(d.coverage.keys(), d.coverage.values())
    #plt.xticks(rotation=90)
    #plt.show()
        