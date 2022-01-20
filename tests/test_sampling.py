from unittest import TestCase

from pyco.modeling import CNFExpression
from pyco.sampling import DistanceBasedSampler
from . import THIS_DIR


class TestDistanceBasedSampler(TestCase):
    def test_sample(self):
        cnf = CNFExpression()
        cnf.from_dimacs(THIS_DIR + "/feature_models/h2.dimacs")

        sampler = DistanceBasedSampler(cnf)
        sample = sampler.sample(1000)
