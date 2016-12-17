from scipy.special import digamma, gammaln
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import time

def convert_doc(doc):
    row = np.array(doc.todense())[0]
    row = np.array([tup[0] for tup in enumerate(row) for _ in range(tup[1])])
    assert (doc.sum() == len(row))
    return row

def remove_word(row):
    index_remove = np.random.choice(range(len(row)))
    word = row[index_remove]
    return np.delete(row, index_remove), word

class LDA:
    def __init__(self, K, D, V, alpha, eta, tau, kappa):
        self.K = K
        self.D = D
        self.V = V
        self.alpha = alpha
        self.eta = eta
        self.tau = tau
        self.kappa = kappa
        self.learning_rate = lambda t : pow(tau+t+1, -kappa)
        self.lmbda = np.random.rand(self.K, self.V)

    def e_step(self, docs, gamma=None, max_iter=100):
        # deal with the case where only 1 document was passed in (rather than a list of documents)
        if type(docs) != list:
            docs = [docs]
        batch_size = len(docs)
        if gamma is None:
            gamma = np.ones((len(docs), self.K))
        phi = [np.zeros((l, self.K)) for l in map(len, docs)]
        digamma_lambda = digamma(self.lmbda)
        digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
        ElogbetaT = digamma_lambda.T - digamma_lambda_sum
        change = 5. * batch_size
        iter_count = 0
        while change / batch_size > 0.0001 and iter_count < max_iter:
            old_phi = [mat.copy() for mat in phi]
            phi = [ElogbetaT[row] + (digamma(gamma_row) - digamma(gamma_row.sum())) for row, gamma_row in zip(docs, gamma)]
            phi = [(mat.T - mat.max(axis=1)).T for mat in phi]
            phi = map(np.exp, phi)
            phi = [(mat.T / mat.sum(axis=1)).T for mat in phi]
            gamma = np.array([mat.sum(axis=0) + self.alpha for mat in phi])
            change = np.sqrt(sum([pow(np.linalg.norm(mat-old_mat), 2) for mat, old_mat in zip(phi, old_phi)]))
            print iter_count, change / batch_size, '\r',
            sys.stdout.flush()
            iter_count += 1
        print
        return phi, gamma

    def m_step(self, docs, phi, t):
        batch_size = len(docs)
        lengths = map(len, docs)
        assert(min(lengths) > 0)
        doc_mats = [np.zeros((l, self.V)) for l in lengths]
        for d in range(batch_size):
            for n in range(lengths[d]):
                doc_mats[d][n][docs[d][n]] = 1.
        # temp_mat == np.array([np.dot(mat.T, doc_mat) for mat, doc_mat in zip(phi, doc_mats)]).sum(axis=0). This loop
        # simply gives a more space-efficient way to compute that sum of matrix products.
        temp_mat = np.zeros((self.K, self.V))
        for mat, doc_mat in zip(phi, doc_mats):
            prod = np.dot(mat.T, doc_mat)
            temp_mat += prod
            del prod
        lmbda_new = float(self.D) / batch_size * temp_mat + self.eta
        self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new

    def batch_update(self, docs, t):
        phi, gamma = self.e_step(docs)
        self.m_step(docs, phi, t)
        return phi, gamma

    def fit_batched(self, X, batch_size=16, n_iter=1000): # TODO create a new version of this that avoids computing anything for the same term multiple times; i.e. if a single term appears in the doc multiple times, then just compute it once. maybe model this after Hoffman et al's code?
        assert(self.V == X.shape[1])
        t = 0
        elbo_lst = []
        while t < n_iter:
            print t, '\r',
            sys.stdout.flush()
            sample_indices = np.random.choice(self.D, size=batch_size, replace=False) # TODO should this be with replacement?
            sample_indices = filter(lambda d : X[d].sum() > 0, sample_indices)
            rows = [convert_doc(X[d]) for d in sample_indices]

            phi, gamma = self.batch_update(rows, t)
            elbo_lst.append(self.elbo(rows, phi, gamma))
            t += 1
            if t % 1000000 == 0:
                sns.heatmap(self.lmbda)
                plt.show()
                sns.heatmap((self.lmbda.T / self.lmbda.sum(axis=1)).T)
                plt.show()
        plt.plot(elbo_lst)
        plt.show()
        elbo_lst = np.array(elbo_lst)
        window = 5
        plt.plot(reduce(lambda a,b : a+b, [elbo_lst[i:len(elbo_lst)-window+i] for i in range(window)]) / window)
        plt.show()

    def elbo(self, docs, phi=None, gamma=None):
        if phi is None or gamma is None:
            phi, gamma = self.e_step(docs)

        digamma_lambda = digamma(self.lmbda)
        digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
        ElogbetaT = digamma_lambda.T - digamma_lambda_sum

        digamma_gamma = digamma(gamma)
        digamma_gamma_sum = digamma(gamma.sum(axis=1))
        ElogthetaT = digamma_gamma.T - digamma_gamma_sum

        # E[log p(beta|eta)]
        Elogpbeta = self.K * (gammaln(self.V * self.eta) - self.V * gammaln(self.eta)) + (self.eta-1) * ElogbetaT.sum()
        # E[log p(theta|alpha)]
        Elogptheta = self.D * (gammaln(self.K * self.alpha) - self.K * gammaln(self.alpha)) + (self.alpha-1) * ElogthetaT.sum()
        # E[log p(z|theta)]
        Elogpz = (np.array([phi_mat.sum(axis=0) for phi_mat in phi]) * ElogthetaT.T).sum()
        ## E[log p(X|beta,theta)], which seems to be what Hoffman et al. 2013 uses
        #ElogpX = 0.
        #for i, doc in enumerate(docs):
        #    ElogpX += ElogthetaT.T[i].sum() * len(doc) + ElogbetaT[doc].sum()
        # E[log p(X|z,beta)]
        ElogpX = 0.
        assert(len(docs)==len(phi))
        for doc, phi_mat in zip(docs, phi):
            #print phi_mat.shape, ElogbetaT[doc].shape
            assert(phi_mat.shape == ElogbetaT[doc].shape)
            ElogpX += (phi_mat * ElogbetaT[doc]).sum()
        # E[log q(beta|lambda)]
        Elogqbeta = ((self.lmbda - 1) * ElogbetaT.T).sum() - gammaln(self.lmbda).sum() + gammaln(self.lmbda.sum(axis=1)).sum()
        # E[log q(theta|gamma)]
        Elogqtheta = ((gamma - 1) * ElogthetaT.T).sum() - gammaln(gamma).sum() + gammaln(gamma.sum(axis=1)).sum()
        # E[log q(z|phi)]
        Elogqz = sum([(phi_mat * np.log(phi_mat)).sum() for phi_mat in phi])
        # compute ELBO, while making sure that local (document-specific) terms are multipled by D/(batch size), to
        # account for the fact that we are only computing the ELBO on a small batch
        score = float(self.D) / len(docs) * (Elogptheta + Elogpz + ElogpX - Elogqtheta - Elogqz) + Elogpbeta - Elogqbeta
        return score
        #return float(self.D) / len(docs) * ElogpX, float(self.D) / len(docs) * (Elogptheta - Elogqtheta), float(self.D) / len(docs) * (Elogpz - Elogqz), Elogpbeta - Elogqbeta, score

    def predictive_log_likelihood(self, gamma, word):
        assert(gamma.shape[0] == self.K)
        assert(self.lmbda.T[word].shape[0] == self.K)
        assert(self.lmbda.sum(axis=1).shape[0] == self.K)
        return np.log((gamma * self.lmbda.T[word] / self.lmbda.sum(axis=1)).sum()) - np.log(gamma.sum())
        # TODO implement

    # TODO implement functions to output the log likelihood or perplexity (esp. for the sake of model comparison with HDPs)
    # TODO look at the ELBO computation in Hoffman et al. 2010 (section 2.1)s, to check the correctness of what I have
