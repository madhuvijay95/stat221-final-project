from scipy.special import digamma
import numpy as np
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDAsklearn
import gensim

class LDA:
    def __init__(self, K, alpha, eta, learning_rate=None):
        self.K = K
        self.alpha = alpha
        self.eta = eta
        self.learning_rate = learning_rate if learning_rate is not None else (lambda t : 1./(t+1))

    # takes a D x V matrix of counts (e.g. output of CountVectorizer) as input
    def fit(self, X, n_iter=1000):
        self.D, self.V = X.shape
        self.lmbda = np.random.rand(self.K, self.V)
        self.beta = np.zeros((self.K, self.V))
        #self.varphi = np.zeros(D, V, K)
        #self.gamma = np.zeros(D, K)
        t = 0
        while t < n_iter:
            d = np.random.randint(0, self.D)
            print t, d, self.beta.sum(), '\r',
            sys.stdout.flush()
            gamma = np.ones(self.K) # K
            delta = digamma(self.lmbda.T) # V x K
            delta = (delta.T - delta.sum(axis=1)).T # V x K
            varphi = np.zeros((self.V, self.K)) # initialize varphi (just for the sake of having a well-defined while-loop condition below)
            change = 5.
            while change > 0.001: # TODO is this a sufficient check of convergence? should it be made more/less stringent?
                old_varphi = varphi.copy()
                varphi = delta + digamma(gamma) # V x K
                varphi = np.exp(varphi) # V x K
                varphi = (varphi.T / varphi.sum(axis=1)).T # V x K
                gamma = np.dot(varphi.T, np.array(X[d].todense())[0]) + self.alpha # K
                change = np.linalg.norm(varphi-old_varphi)
            lmbda_new = self.D * varphi.T * np.array(X[d].todense())[0] + self.eta
            #lmbda_new = self.D * np.dot(varphi.T, np.array(X[d].todense())[0]) + self.eta # wrong update rule -- need an elementwise product
            self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new
            self.beta = self.beta + varphi.T * np.array(X[d].todense())[0]
            if t % 10000 == 0:
                sns.heatmap((self.beta.T / self.beta.sum(axis=1)).T)
                plt.show()
                #self.beta = np.zeros((self.K, self.V)) # TODO remove this
            t += 1
            # TODO maybe have self.varphi be t x V x K instead of D x V x K; just append a V x K matrix in each iteration
        self.beta = (self.beta.T / self.beta.sum(axis=1)).T

with open(sys.path[0] + '\\' + sys.argv[1], 'r') as f:
    corpus = [line[:-1] for line in f.readlines()]
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
print vectorizer.vocabulary_
print X.shape
#sns.heatmap(X.T.dot(X).todense())
#plt.show()

n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
lda = LDA(3, 1, 1, learning_rate = lambda t : 0.1)#lambda t : (t+1)**(-0.51)) # TODO what are correct values of alpha and eta?
lda.fit(X, n_iter=n_iter)
print 'lambda (from my LDA)'
sys.stdout.flush()
sns.heatmap(lda.lmbda)
plt.show()
print 'beta (from my LDA)'
sys.stdout.flush()
sns.heatmap(lda.beta)
plt.show()

lda_sklearn = LDAsklearn(n_topics=3).fit(X)
print 'beta (from sklearn LDA)'
sys.stdout.flush()
sns.heatmap(lda_sklearn.components_)
plt.show()
print dir(lda_sklearn)
X_lda = [list(enumerate(np.array(row.todense())[0])) for row in X]
lda = gensim.models.ldamodel.LdaModel(corpus=X_lda, num_topics=3, id2word={v:k for k,v in vectorizer.vocabulary_.items()})
print lda.print_topics()