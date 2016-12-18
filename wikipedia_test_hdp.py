from lda import LDA, convert_doc, remove_word
from hdp import HDP
from wikirandom import get_random_wikipedia_articles

import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.decomposition import LatentDirichletAllocation as LDAsklearn
#import gensim
import time
import cPickle as pickle
import scipy.sparse

with open(sys.path[0] + '\\dict.txt', 'r') as f:
    vocab_list = [s[:-1] for s in f.readlines()]
vectorizer = CountVectorizer(vocabulary=vocab_list)

V = len(vectorizer.vocabulary)
n_topics = int(sys.argv[1])
n_topics_per_doc = int(sys.argv[2])
batch_size = int(sys.argv[3])
n_iter = int(sys.argv[4])
kappa = float(sys.argv[5]) if len(sys.argv) > 5 else 0.51
D = batch_size * n_iter # is this reasonable?
max_retrieve = 64 # largest number of articles that are queried together in 1 function call
hdp = HDP(n_topics, n_topics_per_doc, D, V, 1., 0.01, 100., 1, kappa)

elbo_lst = []
scrape_time = 0.
examples = []
log_likelihoods = []
start_time_loop = time.time()
for t in range(n_iter):
    print '====================BATCH %d====================' % t
    sys.stdout.flush()
    articlenames = []
    n_requested = 0
    mats = []
    while n_requested < batch_size:
        request_size = min(batch_size-n_requested, max_retrieve)
        start_time = time.time()
        articles_temp, articlenames_temp = get_random_wikipedia_articles(request_size)
        sys.stdout.flush()
        end_time = time.time()
        scrape_time += end_time - start_time

        mat_temp = vectorizer.transform(articles_temp)
        mats.append(mat_temp)

        articlenames.extend(articlenames_temp)
        n_requested += request_size

        del articles_temp, articlenames_temp
    #mat = vectorizer.transform(articles)
    mat = scipy.sparse.vstack(tuple(mats), format='csr')
    mat = mat[filter(lambda d : mat[d].sum() > 1, range(mat.shape[0]))] # check for length > 1 -- need at least 2 words, to have 1 for held-out likelihood and >=1 for training
    rows = [convert_doc(mat[d]) for d in range(mat.shape[0])]
    rows, words = zip(*([remove_word(row) for row in rows]))
    rows = list(rows)
    gamma1, gamma2, zeta, phi = hdp.batch_update(rows, t)
    pred_log_likelihoods = [hdp.predictive_log_likelihood(gamma1_row, gamma2_row, zeta_row, word) for gamma1_row, gamma2_row, zeta_row, word in zip(gamma1, gamma2, zeta, words)]
    print zip([vectorizer.vocabulary[word] for word in words], pred_log_likelihoods)
    print np.mean(pred_log_likelihoods)
    log_likelihoods.append(np.mean(pred_log_likelihoods))
    #extreme_row = np.argmax((gamma.T / gamma.sum(axis=1)).max(axis=0))
    #extreme_row_topic = np.argmax(gamma[extreme_row])
    #examples.append((articlenames[extreme_row], extreme_row_topic, (gamma.T / gamma.sum(axis=1)).max()))
    #elbo_lst.append(hdp.elbo(rows, phi, gamma))
end_time_loop = time.time()
total_time = end_time_loop - start_time_loop
train_time = total_time - scrape_time

print
print
print
print 'Total time for scraping and processing articles: %.3f seconds' % scrape_time
print 'Total time for everything else (converting data and training model): %.3f seconds' % train_time
print

mean_dist = (hdp.lmbda.T / hdp.lmbda.sum(axis=1)).T
mean_dist_normalized = mean_dist - mean_dist.mean(axis=0)
print 'Representative words for each topic:'
for topic, row in enumerate(mean_dist_normalized):
    print '%d:' % topic, [vocab_list[ind] for ind in sorted(range(len(row)), key = lambda ind : -row[ind])[0:20]]
print
print 'Most popular words for each topic:'
for topic, row in enumerate(mean_dist):
    print '%d:' % topic, [vocab_list[ind] for ind in sorted(range(len(row)), key = lambda ind : -row[ind])[0:20]]
print

#print 'Examples of articles from each topic:'
#for topic in set(zip(*examples)[1]):
#    print '%d:' % topic, [tup[0] + ' (%.1f%%)' % (100*tup[2]) for tup in examples if tup[1]==topic]

#if len(log_likelihoods) < 100:
#    print
#    print 'Log likelihoods:', log_likelihoods
#if len(elbo_lst) < 100:
#    print
#    print 'ELBOs:', elbo_lst

sys.stdout.flush()

topic_proportions_cumulative = 1 - np.exp(np.cumsum(np.log(hdp.b / (hdp.a + hdp.b))))
print list(enumerate(topic_proportions_cumulative))
plt.plot(topic_proportions_cumulative)
plt.savefig('wikipedia_HDP_proportions_cumulative_%d_%d_%d_%d_%.2f.png' % (hdp.K, hdp.T, batch_size, n_iter, hdp.kappa))
plt.close()

with open('wikipedia_HDP_log_likelihoods_%d_%d_%d_%d_%.2f.p' % (hdp.K, hdp.T, batch_size, n_iter, hdp.kappa), 'w') as f:
    pickle.dump(log_likelihoods, f)
#with open('wikipedia_elbos_%d_%d_%d_%.2f.p' % (hdp.K, hdp.T, batch_size, hdp.kappa), 'w') as f:
#    pickle.dump(elbo_lst, f)
with open('wikipedia_HDP_details_%d_%d_%d_%d_%.2f.p' % (hdp.K, hdp.T, batch_size, n_iter, hdp.kappa), 'w') as f:
    pickle.dump({'K' : hdp.K, 'T' : hdp.T, 'D' : hdp.D, 'V' : hdp.V, 'alpha' : hdp.alpha, 'eta' : hdp.eta,
                 'omega' : hdp.omega, 'tau' : hdp.tau, 'kappa' : hdp.kappa, 'lambda' : hdp.lmbda, 'a' : hdp.a,
                 'b' : hdp.b, 'scrape_time' : scrape_time, 'train_time' : train_time}, f)

plt.plot(log_likelihoods)
plt.savefig('wikipedia_HDP_log_likelihoods_%d_%d_%d_%d_%.2f.png' % (hdp.K, hdp.T, batch_size, n_iter, hdp.kappa))
plt.close()
#plt.show()
log_likelihoods = np.array(log_likelihoods)
window = 5
if window <= len(log_likelihoods):
    plt.plot(reduce(lambda a,b : a+b, [log_likelihoods[i:len(log_likelihoods)-window+i] for i in range(window)]) / window)
    plt.savefig('wikipedia_HDP_log_likelihoods_moving_avg_%d_%d_%d_%d_%d_%.2f.png' % (window, hdp.K, hdp.T, batch_size, n_iter, hdp.kappa))
    plt.close()
    #plt.show()
#plt.plot(elbo_lst)
#plt.savefig('wikipedia_HDP_elbos_%d_%d_%d_%.2f.png' % (hdp.K, hdp.T, batch_size, hdp.kappa))
#plt.show()
