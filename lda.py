from scipy.special import digamma, gammaln
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDAsklearn
import gensim
import time

def convert_doc(mat, index):
    row = np.array(mat[index].todense())[0]
    row = np.array([tup[0] for tup in enumerate(row) for _ in range(tup[1])])
    assert (mat[index].sum() == len(row))
    return row

class LDA:
    def __init__(self, K, alpha, eta, tau, kappa):
        self.K = K
        self.alpha = alpha # TODO figure out how to set these parameters -- model bsaed on Hoffman et al.'s code
        self.eta = eta
        self.learning_rate = lambda t : pow(tau+t+1, -kappa)

    def e_step(self, docs):
        # deal with the case where only 1 document was passed in (rather than a list of documents)
        if type(docs) != list:
            docs = [docs]
        gamma = np.ones((len(docs), self.K))
        phi = [np.zeros((l, self.K)) for l in map(len, docs)]
        digamma_lambda = digamma(self.lmbda)
        digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
        ElogbetaT = digamma_lambda.T - digamma_lambda_sum
        change = 5.
        while change > 0.0001:
            old_phi = [mat.copy() for mat in phi]
            phi = [ElogbetaT[row] + (digamma(gamma_row) - digamma(gamma_row.sum())) for row, gamma_row in zip(docs, gamma)]
            phi = [(mat.T - mat.max(axis=1)).T for mat in phi]
            phi = map(np.exp, phi)
            phi = [(mat.T / mat.sum(axis=1)).T for mat in phi]
            gamma = np.array([mat.sum(axis=0) + self.alpha for mat in phi])
            change = np.sqrt(sum([pow(np.linalg.norm(mat-old_mat), 2) for mat, old_mat in zip(phi, old_phi)]))
        return phi, gamma

    def fit2_batched(self, X, batch_size=16, n_iter=1000): # TODO create a new version of this that avoids computing anything for the same term multiple times; i.e. if a single term appears in the doc multiple times, then just compute it once. maybe model this after Hoffman et al's code?
        self.D, self.V = X.shape
        self.lmbda = np.random.rand(self.K, self.V)
        t = 0
        elbo_lst = []
        while t < n_iter:
            sample_indices = np.random.choice(self.D, size=batch_size, replace=False) # TODO should this be with replacement?
            sample_indices = filter(lambda d : X[d].sum() > 0, sample_indices)
            curr_batch_size = len(sample_indices)
            print t,
            sys.stdout.flush()
            rows = [convert_doc(X, d) for d in sample_indices]

            lengths = map(len, rows)
            assert(min(lengths) > 0)
            doc_mats = [np.zeros((l, self.V)) for l in lengths]
            for d in range(curr_batch_size):
                for n in range(lengths[d]):
                    doc_mats[d][n][rows[d][n]] = 1.
            ## TODO precompute E[beta|lambda] matrix here, to reduce the number of computations in the inner loop below
            ## TODO split the E-step and M-step into 2 separate functions, and call both functions in fit2()
            phi, gamma = self.e_step(rows)
            lmbda_new = float(self.D) / curr_batch_size * np.array([np.dot(mat.T, doc_mat) for mat, doc_mat in zip(phi, doc_mats)]).sum(axis=0) + self.eta
            self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new
            start_time = time.time()
            elbo_lst.append(self.elbo(rows, phi, gamma))
            end_time = time.time()
            print type(elbo_lst[-1]), '%.3f' % (end_time - start_time), '\r',
            sys.stdout.flush()
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

    # TODO I believe this is now obsolete; use fit2_batched instead
    def fit2(self, X, n_iter=1000): # TODO create a new version of this that avoids computing anything for the same term multiple times; i.e. if a single term appears in the doc multiple times, then just compute it once. maybe model this after Hoffman et al's code?
        self.D, self.V = X.shape
        self.lmbda = np.random.rand(self.K, self.V)
        t = 0
        while t < n_iter:
            d = np.random.randint(0, self.D)
            print t, d, '\r',
            sys.stdout.flush()
            gamma = np.ones(self.K)
            row = convert_doc(X, d)
            N = len(row)
            if N == 0:
                continue
            doc_mat = np.zeros((N, self.V))
            for n in range(N):
                doc_mat[n][row[n]] = 1.
            phi = np.zeros((N, self.K))
            digamma_lambda = digamma(self.lmbda)
            digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
            change = 5.
            while change > 0.0001:
                old_phi = phi.copy()
                phi = digamma_lambda.T[row] + (digamma(gamma) - digamma_lambda_sum - digamma(gamma.sum()))
                phi = (phi.T - phi.max(axis=1)).T
                phi = np.exp(phi)
                phi = (phi.T / phi.sum(axis=1)).T
                gamma = phi.sum(axis=0) + self.alpha
                change = np.linalg.norm(phi-old_phi)
            lmbda_new = self.D * np.dot(phi.T, doc_mat) + self.eta
            self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new
            t += 1
            if t % 1000000 == 0:
                sns.heatmap(self.lmbda)
                plt.show()
                sns.heatmap((self.lmbda.T / self.lmbda.sum(axis=1)).T)
                plt.show()

    def fit_batched(self, X, batch_size=16, n_iter=1000):
        self.D, self.V = X.shape
        self.lmbda = np.random.rand(self.K, self.V)
        self.beta = np.zeros((self.K, self.V))
        t = 0
        while t < n_iter:
            sample_indices = np.random.choice(self.D, size=batch_size, replace=False) # TODO should this be with replacement?
            sample_indices = filter(lambda d : X[d].sum() > 0, sample_indices)
            curr_batch_size = len(sample_indices)
            print t, '\r',
            sys.stdout.flush()
            gamma = np.ones((curr_batch_size, self.K)) # K
            digamma_lambda = digamma(self.lmbda) # K x V
            digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
            varphi = np.zeros((curr_batch_size, self.V, self.K)) # initialize varphi (just for the sake of having a well-defined while-loop condition below)
            change = 5.
            while change > 0.0001:
                old_varphi = varphi.copy()
                #print np.array([np.tile(arr, (self.V, 1)) for arr in digamma(gamma)]).shape, digamma_lambda.T.shape, digamma_lambda_sum.shape, digamma(gamma.sum()).shape
                varphi = np.array([np.tile(arr, (self.V, 1)) for arr in digamma(gamma)]) + (digamma_lambda.T - (digamma_lambda_sum + digamma(gamma.sum()))) # B x V x K
                varphi = varphi.T # K x V x B
                varphi = varphi - varphi.max(axis=0) # K x V x B
                varphi = np.exp(varphi) # K x V x B
                varphi = varphi / varphi.sum(axis=0) # K x V x B
                varphi = varphi.T # B x V x K
                gamma = np.array([np.dot(varphi_mat.T, doc_vec) for varphi_mat, doc_vec in zip(varphi, np.array(X[sample_indices].todense()))]) + self.alpha # B x K
                change = np.linalg.norm(varphi-old_varphi)
            lmbda_new = float(self.D) / curr_batch_size * (varphi.T * np.array(X[sample_indices].T.todense())).sum(axis=2) + self.eta
            self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new
            t += 1
            if t % 1000 == 0:
                sns.heatmap(self.lmbda)
                plt.show()


    # takes a D x V matrix of counts (e.g. output of CountVectorizer) as input
    def fit(self, X, n_iter=1000): # TODO make a batched version of this as well
        self.D, self.V = X.shape
        self.lmbda = np.random.rand(self.K, self.V)
        self.beta = np.zeros((self.K, self.V))
        t = 0
        while t < n_iter:
            d = np.random.randint(0, self.D)
            print t, d, self.beta.sum(), '\r',
            sys.stdout.flush()
            gamma = np.ones(self.K) # K
            digamma_lambda = digamma(self.lmbda) # K x V
            digamma_lambda_sum = digamma(self.lmbda.sum(axis=1))
            varphi = np.zeros((self.V, self.K)) # initialize varphi (just for the sake of having a well-defined while-loop condition below)
            change = 5.
            while change > 0.001: # TODO is this a sufficient check of convergence? should it be made more/less stringent?
                old_varphi = varphi.copy()
                varphi = digamma_lambda.T + (digamma(gamma) - digamma_lambda_sum - digamma(gamma.sum())) # V x K
                varphi = (varphi.T - varphi.max(axis=1)).T # subtract the max element from each log distribution -- ensures numerical stability in exponentiation and doesn't affect final result
                #varphi = varphi - varphi.max(axis=0) # subtract the max element from each log distribution -- ensures numerical stability in exponentiation and doesn't affect final result
                varphi = np.exp(varphi)
                #####varphi = varphi / varphi.sum(axis=0) # TODO figure this out -- am I normalizing correctly? also might need to fix the max-subtraction stuff above, based on that
                varphi = (varphi.T / varphi.sum(axis=1)).T # V x K
                gamma = np.dot(varphi.T, np.array(X[d].todense())[0]) + self.alpha # K
                change = np.linalg.norm(varphi-old_varphi)
            lmbda_new = self.D * varphi.T * np.array(X[d].todense())[0] + self.eta
            #lmbda_new = self.D * np.dot(varphi.T, np.array(X[d].todense())[0]) + self.eta # wrong update rule -- need an elementwise product
            self.lmbda = (1 - self.learning_rate(t)) * self.lmbda + self.learning_rate(t) * lmbda_new
            self.beta = self.beta + varphi.T * np.array(X[d].todense())[0]
            t += 1
            if t % 1000 == 0:
                sns.heatmap(self.beta)
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

    # TODO implement functions to output the log likelihood or perplexity (esp. for the sake of model comparison with HDPs)
    # TODO look at the ELBO computation in Hoffman et al. 2010 (section 2.1)s, to check the correctness of what I have

with open(sys.path[0] + '\\' + sys.argv[1], 'r') as f:
    corpus = [line[:-1] for line in f.readlines()]
vectorizer = CountVectorizer(stop_words='english', min_df=10) # *****TODO can't just CountVectorizer the whole input, because that's a non-online operation. we need to pre-specify a vocabulary (e.g. use the vocabulary they used in Hoffman et al.'s code), so that it can be done in a streaming/online way without having to infer the vocab from the whole dataset
X = vectorizer.fit_transform(corpus)
print len(vectorizer.vocabulary_)
print X.shape
vocab_list = sorted(vectorizer.vocabulary_, key = lambda word : vectorizer.vocabulary_[word])

n_topics = int(sys.argv[2])
n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
lda = LDA(n_topics, 1./n_topics, 1./n_topics, 1, 0.51)
start_time = time.time()
lda.fit2_batched(X, n_iter=n_iter)
end_time = time.time()
print
print 'Total time to fit LDA model: %.3f seconds' % (end_time - start_time)
sys.stdout.flush()
mean_dist = (lda.lmbda.T / lda.lmbda.sum(axis=1)).T
mean_dist_normalized = mean_dist - mean_dist.mean(axis=0)
for row in mean_dist_normalized:
    print [vocab_list[ind] for ind in sorted(range(len(row)), key = lambda ind : -row[ind])[0:20]]
    sys.stdout.flush()
#print 'lambda (from my LDA):', lda.lmbda
#sys.stdout.flush()
#sns.heatmap(lda.lmbda)
#plt.show()
#sns.heatmap((lda.lmbda.T / lda.lmbda.sum(axis=1)).T)
#plt.show()
#print 'beta (from my LDA):', lda.beta
#sys.stdout.flush()
#sns.heatmap(lda.beta)
#plt.show()

print
print

#lda_sklearn = LDAsklearn(n_topics=n_topics).fit(X)
#for row in lda_sklearn.components_ - lda_sklearn.components_.mean(axis=0):
#    print [vocab_list[ind] for ind in sorted(range(len(row)), key = lambda ind : -row[ind])[0:20]]
#    sys.stdout.flush()
#print 'beta (from sklearn LDA)'
#sys.stdout.flush()
#sns.heatmap(lda_sklearn.components_)
#plt.show()

print
print

X_lda = [list(enumerate(np.array(row.todense())[0])) for row in X]
start_time = time.time()
lda = gensim.models.ldamodel.LdaModel(corpus=X_lda, num_topics=n_topics, id2word={v:k for k,v in vectorizer.vocabulary_.items()})
end_time = time.time()
print 'Total time to fit gensim LDA model: %.3f seconds' % (end_time - start_time)
sys.stdout.flush()
topics = lda.print_topics(num_words=20) # TODO this doesn't match the output of the above. I should be printing the "representative" words for each topic, not the most popular ones -- e.g. "said" is the top word in every single topic right now.
print topics
for topic in topics:
    print [word[6:] for word in topic.split(' + ')]