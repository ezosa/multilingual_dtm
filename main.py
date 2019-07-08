import pickle
from ml_dtm import ML_DTM
from corpus import get_denews_corpus, get_yle_corpus, prune_vocabulary, sample_yle_articles

################## train on DE-News corpus ##################
n_topics = 10
alpha = 0.5
beta = 0.3
psi = 1.0
sigma = 1.0
iterations = 400
vocab_len = 5000
denews_path = "data/news8/"

documents, timestamps, dictionary = get_denews_corpus(denews_path)
documents, dictionary = prune_vocabulary(documents, vocab_len=vocab_len)
model = ML_DTM(documents, dictionary, alpha, beta, psi, sigma, n_topics, iterations)
model.gibbs_sampling()

model_filename = "trained_models/mldtm/mldtm_denews"
f = open(model_filename+".pkl", "wb")
pickle.dump(model, f)
f.close()

################## train on YLE corpus ##################
n_topics = 10
alpha = 0.5
beta = 0.5
psi = 1.0
sigma = 1.0
iterations = 600
vocab_len = 5000
n_timeslices = 10
max_doc = 500

documents, timestamps, dictionary = get_yle_corpus(n_timeslices)
documents = sample_yle_articles(documents, max_doc)
documents, dictionary = prune_vocabulary(documents, vocab_len=vocab_len)
model = ML_DTM(documents, dictionary, alpha, beta, psi, sigma, n_topics, iterations)
model.gibbs_sampling()

model_filename = "trained_models/mldtm/mldtm_yle"
f = open(model_filename+".pkl", "wb")
pickle.dump(model, f)
f.close()

