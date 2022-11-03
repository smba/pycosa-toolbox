import numpy as np
import pycosa.modeling as modeling


class SyntheticDataGenerator:
    def __init__(
        self,
        n_options: int = 5,
        p_factors: float = 0.2,
        p_terms: float = 0.8,
        p_alphas: float = 0.001,
        noise: float = 0.3,
    ):
        """
        Generate a ground-truth performance model that predicts the performance of arbitrary configurations.

        The performance model has five free parameters:

        1) Number of options
        2) Percentage of influential factors (terms, i.e. a single configuration option or combination)

        3) Distribution of  number of options per factor (geometric distribution)
        4) Distribution of influences (equally-sized vs. few large, many small influences): Dirichlet distribution
        5) Noise factor (normal distribution)
        """

        _options = np.arange(n_options)
        self.n_options = n_options

        # sample the number of influential terms
        n_terms = max(1, int(p_factors * n_options))

        # Size (number of options) for each term
        n_factors = np.random.geometric(p_terms, size=n_terms)
        n_factors = [min(nf, n_options) for nf in n_factors]

        terms = [
            np.random.choice(_options, size=nf, replace=False).tolist()
            for nf in n_factors
        ]

        self.terms = terms
        # estimate the alphas for dirichlet
        alphas = np.random.geometric(p_alphas, size=n_terms)

        # sample how much variance a term explains
        variance = np.random.dirichlet(alphas, 1)
        influence = variance * 1000

        # p.linspace(0.01, 0.15, 5)
        influence *= np.random.choice([-1, 1], size=influence.shape)
        influence = influence[0]
        self.influence = influence

        self.model = lambda x: np.sum(
            [
                np.all([x[i] for i in term]) * influence[j]
                for j, term in enumerate(terms)
            ]
        ) + np.random.normal(0, noise)

    def get_performance(self, x) -> float:
        return self.model(x)

    def get_coefs(self):
        return list(zip(self.terms, self.influence))


class AttributedVariabilityModelGenerator:
    def __init__(self, vm: modeling.VariabilityModel):
        self.vm = vm
        self.performance_model = SyntheticDataGenerator(
            n_options=len(self.vm.get_binary_features()), p_factors=0.3, p_terms=0.7
        )

    def get_performances(self, X):
        perf = [self.performance_model.get_performance(x) for x in X]
        return np.array(perf)


if __name__ == "__main__":
    pass
