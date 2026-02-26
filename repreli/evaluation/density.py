
from sklearn.mixture import GaussianMixture

from vonMF.von_mises_fisher_mixture import VonMisesFisherMixture


class ProbabilityDensity:
    def __init__(self, method='movMF', param=20):
        self.method = method
        self.param = param

        if self.method == 'movMF':
            self.kernel = VonMisesFisherMixture(n_clusters=self.param, posterior_type='soft', max_iter=30, normalize=False)
        elif self.method == 'gmm':
            self.kernel = GaussianMixture(n_components=self.param, max_iter=30)
        else:
            raise NotImplementedError('invalid method')

    def fit(self, rep_ref):
        self.kernel.fit(rep_ref)

    def predict(self, rep_test):
        if self.method == 'movMF':
            scores = self.kernel.predict_log_proba(rep_test)
        elif self.method == 'gmm':
            scores = self.kernel.score_samples(rep_test)
        else:
            print('invalid method')
            scores = None
        return scores
