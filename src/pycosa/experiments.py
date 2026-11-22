"""
Generators of synthetic variability models and ground-truth performance
models for controlled experiments.

:class:`SyntheticDataGenerator` creates a random ground-truth performance
model over a number of options, whose terms and influences are known.
Measured performances can then be compared against the ground truth.
:class:`AttributedVariabilityModelGenerator` attaches such a performance
model to an existing :class:`pycosa.modeling.VariabilityModel`.
"""

import numpy as np

import pycosa.modeling as modeling


class SyntheticDataGenerator:
    """
    A ground-truth performance model that predicts the performance of
    arbitrary configurations.

    The performance model has five free parameters:

    1) Number of options
    2) Percentage of influential factors (terms, i.e., a single
       configuration option or a combination of options)
    3) Distribution of the number of options per factor (geometric
       distribution)
    4) Distribution of influences (equally-sized vs. few large, many small
       influences): Dirichlet distribution
    5) Noise factor (normal distribution)

    A performance value is the sum of the influences of all terms whose
    options are all selected, plus normal noise.
    """

    def __init__(
        self,
        n_options: int = 5,
        p_factors: float = 0.2,
        p_terms: float = 0.8,
        p_alphas: float = 0.001,
        noise: float = 0.3,
        seed=None,
    ):
        """
        Parameters
        ----------
        n_options : int, optional
            Number of configuration options (default 5).
        p_factors : float, optional
            Ratio determining the number of influential terms, roughly
            ``p_factors * n_options`` (default 0.2).
        p_terms : float, optional
            Parameter of the geometric distribution of the number of
            options per term; smaller values yield larger terms
            (default 0.8).
        p_alphas : float, optional
            Parameter of the geometric distribution of the Dirichlet
            alphas; smaller values yield a stronger concentration of the
            total influence in few terms (default 0.001).
        noise : float, optional
            Standard deviation of the normal noise added to each
            measurement (default 0.3).
        seed : int, optional
            Seed for reproducible models.
        """

        rng = np.random.default_rng(seed)
        self._rng = rng
        self.noise = noise

        _options = np.arange(n_options)
        self.n_options = n_options

        # sample the number of influential terms
        n_terms = max(1, int(p_factors * n_options))

        # Size (number of options) for each term
        n_factors = rng.geometric(p_terms, size=n_terms)
        n_factors = [min(nf, n_options) for nf in n_factors]

        terms = [
            rng.choice(_options, size=nf, replace=False).tolist() for nf in n_factors
        ]

        self.terms = terms
        # estimate the alphas for dirichlet
        alphas = rng.geometric(p_alphas, size=n_terms)

        # sample how much variance a term explains
        variance = rng.dirichlet(alphas, 1)
        influence = variance * 1000

        influence *= rng.choice([-1, 1], size=influence.shape)
        influence = influence[0]
        self.influence = influence

    def get_performance(self, x) -> float:
        """
        Predicts the performance of a single configuration.

        Parameters
        ----------
        x : array-like of bool
            One configuration, with one entry per option in index order.

        Returns
        -------
        float
            The (noisy) predicted performance.
        """
        x = np.asarray(x)
        value = sum(
            float(np.all(x[term])) * self.influence[j]
            for j, term in enumerate(self.terms)
        )
        return value + self._rng.normal(0, self.noise)

    def get_performances(self, X) -> np.ndarray:
        """
        Vectorized version of :meth:`get_performance` for a 2D array of
        configurations (one configuration per row).

        Parameters
        ----------
        X : array-like
            2D array of configurations, one row per configuration.

        Returns
        -------
        numpy.ndarray
            The (noisy) predicted performances.
        """
        X = np.asarray(X, dtype=bool)
        value = np.zeros(X.shape[0])
        for term, influence in zip(self.terms, self.influence):
            value += X[:, term].all(axis=1) * influence
        return value + self._rng.normal(0, self.noise, size=X.shape[0])

    def get_coefs(self):
        """
        Returns the ground truth: one entry ``(term, influence)`` per
        influential term.
        """
        return list(zip(self.terms, self.influence))


class AttributedVariabilityModelGenerator:
    """
    Attaches a synthetic ground-truth performance model to an existing
    variability model, which turns configurations into measurable systems.
    """

    def __init__(self, vm: modeling.VariabilityModel, seed=None):
        """
        Parameters
        ----------
        vm : pycosa.modeling.VariabilityModel
            The variability model whose options the performance model is
            built over.
        seed : int, optional
            Seed for reproducible performance models.
        """
        self.vm = vm
        self.performance_model = SyntheticDataGenerator(
            n_options=len(self.vm.get_binary_features()),
            p_factors=0.3,
            p_terms=0.7,
            seed=seed,
        )

    def get_performances(self, X):
        """
        Predicts the performances of the given configurations (one row per
        configuration; columns in the order of ``vm.get_binary_features()``).
        """
        return self.performance_model.get_performances(np.asarray(X))


if __name__ == "__main__":
    pass
