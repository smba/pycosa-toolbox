import unittest

import pycosa.modeling as modeling
import pycosa.sampling as sampling

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
            'OPTIMIZE_DISTINCT',
            'OPTIMIZE_IN_SELECT',
            'IGNORE_CATALOGS',
            'COMPRESS'
        ]
        en, dis = sampler.sample(
            options=options,
            size=n,
        )

        for opt in options:
            self.assertTrue(
                en[opt].sum() == 30
            )
            self.assertTrue(
                dis[opt].sum() == 0
            )

if __name__ == "__main__":
    unittest.main()
