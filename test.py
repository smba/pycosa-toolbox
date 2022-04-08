import unittest

import pycosa.modeling as modeling
import pycosa.sampling as sampling

import matplotlib.pyplot as pltn

import pandas as pd


class TesBDDSamplerSampler(unittest.TestCase):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.BDDSampler(self.fm)
        sampler.constrain_enabled(["COMPRESS"])
        sampler.constrain_disabled(["DEFRAG_ALWAYS"])
        sample = sampler.sample(300)
        # print(sample)
        self.assertTrue(sample["DEFRAG_ALWAYS"].values.sum() == 0)
        self.assertTrue(sample["COMPRESS"].values.sum() == sample.shape[0])


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
