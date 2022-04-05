import unittest

import pycosa.modeling as modeling
import pycosa.sampling as sampling

import matplotlib.pyplot as pltn

import pandas as pd


class TestBDDSampler(unittest.TestCase):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        
        sampler = sampling.BDDSampler(self.fm)
        sampler.constrain_disabled(['COMPRESS'])
        sample = sampler.sample(30)
        #print(sample['COMPRESS'].values*1)


class TesDFSSamplerSampler(unittest.TestCase):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.DFSSampler(self.fm)
        sampler.constrain_disabled(['COMPRESS'])
        sample = sampler.sample(30)
        self.assertTrue(sample['COMPRESS'].values.sum() == 0)


class TestElementaryEffectSampler(unittest.TestCase):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.ElementaryEffectSampler(self.fm)
        en, dis = sampler.sample(
            ["OPTIMIZE_TWO_EQUALS", "DEFRAG_ALWAYS", "COMPRESS", "RECOMPILE_ALWAYS"], 50
        )
        
        print(en.shape)


if __name__ == "__main__":
    unittest.main()
