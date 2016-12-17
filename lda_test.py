from lda import LDA
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDAsklearn
import gensim
import time

with open(sys.path[0] + '\\dict.txt', 'r') as f:
    vocab_list = [s[:-1] for s in f.readlines()]
vectorizer = CountVectorizer(vocabulary=vocab_list)

with open(sys.path[0] + '\\' + sys.argv[1], 'r') as f:
    corpus = [line[:-1] for line in f.readlines()]

X = vectorizer.fit_transform(corpus)
print len(vectorizer.vocabulary_)
print X.shape
vocab_list = sorted(vectorizer.vocabulary_, key = lambda word : vectorizer.vocabulary_[word])
D, V = X.shape

n_topics = int(sys.argv[2])
n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
lda = LDA(n_topics, D, V, 1./n_topics, 1./n_topics, 1, 0.51)
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