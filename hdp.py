from scipy.special import digamma, gammaln
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import time

from lda import convert_doc, remove_word

class HDP:
    def __init__(self):
        # TODO implement
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

        phi = [np.dot(ElogbetaT[row], zeta_mat) for row, zeta_mat in zip(docs, zeta)]
        for row, mat in zip(docs, phi):
            assert (phi.shape == (len(row), self.T))

        return zeta, phi, ElogbetaT

    def e_step(self, docs, phi, ElogbetaT=None):
        batch_size = len(phi)
        if ElogbetaT is None:
            digamma_lambda = digamma(self.lmbda)
            digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
            ElogbetaT = digamma_lambda.T - digamma_lambda_sum

        ElogV = digamma(self.a) - digamma(self.a + self.b)
        ElogoneminusV = digamma(self.b) - digamma(self.a + self.b)
        ElogsigmaV = ElogV + np.append(np.array([0]), np.cumsum(ElogoneminusV)[0:-1])

        change = 5.
        while change > 0.0001:
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
            phi = [(mat / mat.sum(axis=1)).T for mat in phi]

            change = np.sqrt(sum([pow(np.linalg.norm(mat-old_mat), 2) for mat, old_mat in zip(phi, old_phi)]))
            print change

        return gamma1, gamma2, zeta, phi

    def m_step(self):
        # TODO implement
        pass
