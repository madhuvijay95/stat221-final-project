from scipy.special import digamma, gammaln
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import time

from lda import convert_doc, remove_word

class HDP:
    def __init__(self, K, T, D, V, alpha, eta, omega, tau, kappa):
        self.K = K
        self.T = T
        self.D = D
        self.V = V
        self.alpha = alpha
        self.eta = eta
        self.omega = omega
        self.tau = tau
        self.kappa = kappa
        self.learning_rate = lambda t : pow(tau+t+1, -kappa)
        self.lmbda = self.m_lambda = 1.0/self.V + 0.01 * np.random.gamma(1.0, 1.0, (self.K, self.V))
        self.a = np.ones(K)
        self.b = np.ones(K) * omega
        pass

    def init_iter(self, docs, ElogbetaT=None):
        batch_size = len(docs)
        if ElogbetaT is None:
            digamma_lambda = digamma(self.lmbda)
            digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
            ElogbetaT = digamma_lambda.T - digamma_lambda_sum

        zeta_dists = [ElogbetaT[row].sum(axis=0) for row in docs]
        zeta_dists = [np.exp(dist - dist.max()) for dist in zeta_dists]
        zeta = np.array([np.tile(dist / dist.sum(), (self.T, 1)) for dist in zeta_dists])
        assert (zeta.shape == (batch_size, self.T, self.K))

        phi = [np.dot(ElogbetaT[row], zeta_mat.T) for row, zeta_mat in zip(docs, zeta)]
        for row, mat in zip(docs, phi):
            assert (mat.shape == (len(row), self.T))

        return zeta, phi, ElogbetaT

    def e_step(self, docs, phi, ElogbetaT=None, max_iter=100):
        batch_size = len(docs)
        if ElogbetaT is None:
            digamma_lambda = digamma(self.lmbda)
            digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
            ElogbetaT = digamma_lambda.T - digamma_lambda_sum

        ElogV = digamma(self.a) - digamma(self.a + self.b)
        ElogoneminusV = digamma(self.b) - digamma(self.a + self.b)
        ElogsigmaV = ElogV + np.append(np.array([0]), np.cumsum(ElogoneminusV)[0:-1])

        change = 5. * batch_size
        iter_count = 0
        while change / batch_size > 0.0001 and iter_count < max_iter:
            old_phi = [mat.copy() for mat in phi]

            gamma1 = np.array([phi_mat.sum(axis=0) + 1 for phi_mat in phi])
            assert(gamma1.shape == (batch_size, self.T))

            phi_horiz_sums = [phi_mat.sum(axis=1) for phi_mat in phi]
            phi_rev_cumsums = [-np.cumsum(phi_mat, axis=1).T + phi_mat_horiz_sum for phi_mat, phi_mat_horiz_sum in zip(phi, phi_horiz_sums)]
            gamma2 = np.array([mat.sum(axis=1) + self.alpha for mat in phi_rev_cumsums])
            assert(gamma2.shape == (batch_size, self.T))

            zeta = np.array([np.dot(phi_mat.T, ElogbetaT[doc]) for doc, phi_mat in zip(docs, phi)])
            zeta = zeta + ElogsigmaV
            assert (zeta.shape == (batch_size, self.T, self.K))
            zeta = zeta.T
            zeta = zeta - zeta.max(axis=0)
            zeta = np.exp(zeta)
            zeta = zeta / zeta.sum(axis=0)
            zeta = zeta.T

            Elogpi = digamma(gamma1) - digamma(gamma1 + gamma2)
            Elogoneminuspi = digamma(gamma2) - digamma(gamma1 + gamma2)
            Elogsigmapi = Elogpi + np.hstack((np.zeros((batch_size, 1)), np.cumsum(Elogoneminuspi, axis=1).T[0:-1].T))

            phi = [np.dot(ElogbetaT[doc], zeta_mat.T) for doc, zeta_mat in zip(docs, zeta)]
            phi = [phi_mat + Elogsigmapi_mat for phi_mat, Elogsigmapi_mat in zip(phi, Elogsigmapi)]
            phi = [np.exp(mat.T - mat.max(axis=1)) for mat in phi]
            phi = [(mat / mat.sum(axis=0)).T for mat in phi]

            change = np.sqrt(sum([pow(np.linalg.norm(mat-old_mat), 2) for mat, old_mat in zip(phi, old_phi)]))
            print iter_count, change / batch_size, '\r',
            sys.stdout.flush()
            iter_count += 1
        print

        return gamma1, gamma2, zeta, phi

    def m_step(self, docs, zeta, phi, t):
        batch_size = len(docs)
        lengths = map(len, docs)
        assert(min(lengths) > 0)
        doc_mats = [np.zeros((l, self.V)) for l in lengths]
        for d in range(batch_size):
            for n in range(lengths[d]):
                doc_mats[d][n][docs[d][n]] = 1.

        temp_mat = np.array([np.dot(zeta_mat.T, np.dot(phi_mat.T, doc_mat)) for zeta_mat, phi_mat, doc_mat in zip(zeta, phi, doc_mats)]).sum(axis=0)
        lmbda_new = float(self.D) / batch_size * temp_mat + self.eta
        self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new

        a_new = float(self.D) / batch_size * np.array([zeta_mat.sum(axis=0) for zeta_mat in zeta]).sum(axis=0) + 1
        self.a = (1 - self.learning_rate(t)) * self.a + self.learning_rate(t) * a_new

        b_new = float(self.D) / batch_size * np.array([(-np.cumsum(zeta_mat, axis=1).T + zeta_mat.sum(axis=1)).sum(axis=1) for zeta_mat in zeta]).sum(axis=0) + self.omega
        self.b = (1 - self.learning_rate(t)) * self.b + self.learning_rate(t) * b_new

    def batch_update(self, docs, t):
        zeta, phi, ElogbetaT = self.init_iter(docs)
        gamma1, gamma2, zeta, phi = self.e_step(docs, phi, ElogbetaT)
        self.m_step(docs, zeta, phi, t)
        return gamma1, gamma2, zeta, phi

    def predictive_log_likelihood(self, gamma1, gamma2, zeta, word):
        Epi = gamma1 / (gamma1 + gamma2)
        Esigmapi = np.exp(np.append(np.array([0]), np.cumsum(np.log(1 - Epi))[0:-1])) * Epi
        #if (Esigmapi * np.dot(zeta, self.lmbda.T[word])).sum() > 5:
        #    print word
        #    print self.lmbda.T[word]
        #print
        return np.log((Esigmapi * np.dot(zeta, self.lmbda.T[word] / self.lmbda.sum(axis=1))).sum())
