import unittest

import pycosa.modeling as modeling
import pycosa.sampling as sampling

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

"""
Testing should include aspects like
- Sample size
    - negative
    - exact
    - maximum not exceeded
- Uniqueness
"""


class TestSingleSampler(unittest.TestCase):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")


class TestBDDSamplerSampler(TestSingleSampler):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.BDDSampler(self.fm)
        sampler.constrain_enabled(["COMPRESS"])
        sampler.constrain_disabled(["DEFRAG_ALWAYS"])
        sample = sampler.sample(size=300)
        # print(sample)
        self.assertTrue(sample["DEFRAG_ALWAYS"].values.sum() == 0)
        self.assertTrue(sample["COMPRESS"].values.sum() == sample.shape[0])


class TestDistanceBasedSampler(TestSingleSampler):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.DistanceBasedSampler(self.fm)
        sample = sampler.sample(size=50)


class TestCoverageSampler(TestSingleSampler):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.CoverageSampler(self.fm)
        sample = sampler.sample(t=1)
        sample = sampler.sample(t=2)
        sample = sampler.sample(t=1, negwise=True)
        sample = sampler.sample(t=2, negwise=True)


class TestDFSSampler(TestSingleSampler):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.DFSSampler(self.fm)
        sample = sampler.sample(size=100)


class TestDiversitySampler(TestSingleSampler):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.DiversityPromotionSampler(self.fm)
        sample = sampler.sample(size=100)


class TestElementaryEffectSampler(unittest.TestCase):
    def setUp(self):
        self.fm = modeling.CNFExpression()
        self.fm.from_dimacs("_test_data/feature_models/h2.dimacs")

    def test_sample(self):
        sampler = sampling.ElementaryEffectSampler(self.fm)
        n = 30
        options = [
            "OPTIMIZE_DISTINCT",
            "OPTIMIZE_IN_SELECT",
            "IGNORE_CATALOGS",
            "COMPRESS",
        ]
        en, dis = sampler.sample(
            options=options,
            size=n,
        )

        for opt in options:
            self.assertTrue(en[opt].sum() == 30)
            self.assertTrue(dis[opt].sum() == 0)


class TestOfflineSampler(unittest.TestCase):
    def setUp(self):

        N = 10
        options = np.arange(N)
        self.options = options

        np.random.seed(1)
        df = np.random.choice([0, 1], size=(10000, N))
        df = np.unique(df, axis=0)

        df = pd.DataFrame(df, columns=options)
        self.df = df

    def test_ee_sampling_optionwise(self):
        sampler = sampling.OfflineSampler(self.df)

        # feature wise
        for o in self.options:
            options = [o]
            en, dis = sampler.elementary_effect_sample(options, size=5)

            # show that the indices are different
            self.assertTrue(len(en) == len(dis))
            self.assertTrue(en != dis, "Indexes are identical!")
            for i in range(len(en)):
                self.assertTrue(en[i] != dis[i], "{} != {}".format(en[i], dis[i]))

            # show that option $o is different
            for i in range(len(en)):
                cfg1 = self.df.loc[en[i]]
                cfg2 = self.df.loc[dis[i]]
                self.assertTrue(cfg1[o] != cfg2[o])

                # check that configurations only differ in exactly one column
                distance = np.count_nonzero(cfg1 != cfg2)
                self.assertTrue(distance == 1)

    def test_ee_sampling_pairwise(self):
        sampler = sampling.OfflineSampler(self.df)

        # pairwise
        for opt in itertools.combinations(self.options, 2):
            options = list(opt)
            en, dis = sampler.elementary_effect_sample(options, size=5)

            # show that the indices are different
            self.assertTrue(len(en) == len(dis))
            self.assertTrue(en != dis, "Indexes are identical!")
            for i in range(len(en)):
                self.assertTrue(en[i] != dis[i], "{} != {}".format(en[i], dis[i]))

            # show that options $o are different
            for i in range(len(en)):
                cfg1 = self.df.loc[en[i]]
                cfg2 = self.df.loc[dis[i]]
                for o in options:
                    self.assertTrue(cfg1[o] != cfg2[o])

                # check that configurations only differ in exactly one column
                distance = np.count_nonzero(cfg1 != cfg2)
                self.assertTrue(distance == 2)


if __name__ == "__main__":
    unittest.main()
