import numpy as np
import sys

n_topics = 3
n_words_per_topic = 5
n_docs = 1000
doc_length = 1000

topics = np.random.randint(0, n_topics, n_docs)
word_indices = np.random.randint(0, n_words_per_topic, (n_docs, doc_length))
word_strs = map(lambda (topic, words) : ['topic%d_word%d' % (topic, word) for word in words], zip(topics, word_indices))
corpus = map(lambda lst : reduce(lambda w1,w2 : w1+' '+w2, lst) + '\n', word_strs)
with open(sys.path[0] + '\\' + sys.argv[1], 'w') as f:
    f.writelines(corpus)