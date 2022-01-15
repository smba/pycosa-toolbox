import unittest

from pyco.modeling import CNFExpression
from . import THIS_DIR

"""
each sampling strategy is tested for
- should result in unique set of configurations with N samples
- should be deterministic via seed (two repetitions yield same sample)
- should perform consistently on randomized data sets (feature models)

"""


class TestSampler(unittest.TestCase):
    def setUp(self):
        cnf = CNFExpression()
        cnf.from_dimacs(THIS_DIR + "/feature_models/h2.dimacs")

    def test_bener(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()