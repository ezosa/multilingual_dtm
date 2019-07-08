import numpy as np
from time import time
from scipy.stats import entropy
from threading import Thread
import random


class ML_DTM(object):

	def __init__(self, documents, dictionary, alpha=1.0, beta=0.5, psi=1.0, sigma=1.0, n_topics=10, n_iter=1000):
		print("- initializing parameters -")
		self.n_iterations = n_iter
		self.languages = list(documents.keys())
		self.K = n_topics
		self.beta = beta
		self.psi = psi
		self.sigma = sigma
		self.timeslices = [len(documents[self.languages[0]][i]) for i in range(len(documents[self.languages[0]]))]
		self.T = len(self.timeslices)
		self.D = np.sum(self.timeslices)
		self.V = {lang: len(dictionary[lang]) for lang in self.languages}
		self.N = {lang: np.array([[len(doc) for doc in documents[lang][t]] for t in range(self.T)]) for lang in self.languages}
		self.alpha = np.array([[alpha for k in range(self.K)] for t in range(self.T)])
		self.word_id = {lang: {dictionary[lang][i]: i for i in range(len(dictionary[lang]))} for lang in self.languages}
		self.word_token = {lang: dictionary[lang] for lang in self.languages}
		self.z = {lang: [] for lang in self.languages}
		self.w = {lang: [] for lang in self.languages}
		for t in range(self.T):
			n_dt = self.timeslices[t]
			for lang in self.languages:
				self.z[lang].append([[random.randrange(0, self.K) for word in range(self.N[lang][t][d])] for d in range(n_dt)])
				self.w[lang].append([[self.word_id[lang][documents[lang][t][d][word]] for word in range(self.N[lang][t][d])] for d in range(n_dt)])
		# do the counting and compute theta and phi based on the counts
		m, n, n_sum = self.calculate_counts()
		self.counts = {}
		self.counts['m'] = m
		self.counts['n'] = n
		self.counts['n_sum'] = n_sum
		theta, phi = self.compute_theta_phi()
		self.theta = theta
		self.phi = phi
		for lang in self.languages:
			print("Vocabulary size -", lang, ":", str(self.V[lang]))
		print("Topics: ", par['K'])
		print("beta: ", par['beta'])
		print("timeslices:", par['timeslices'])

	def compute_jsd(self,p, q):
		p = np.asarray(p)
		q = np.asarray(q)
		p /= p.sum()
		q /= q.sum()
		m = (p + q) / 2
		return (entropy(p, m) + entropy(q, m)) / 2

	def evaluate_divergence(self):
		topic_div = {lang: [[0.0 for _ in range(self.K-1)] for _ in range(self.T)] for lang in self.languages}
		time_div = {lang: [[0.0 for _ in range(self.T-1)] for _ in range(self.K)] for lang in self.languages}
		for t in range(self.T):
			for k in range(self.K-1):
				for lang in self.languages:
					topic1 = softmax(self.phi[lang][t][k])
					topic2 = softmax(self.phi[lang][t][k+1])
					jsd = self.compute_jsd(topic1, topic2)
					topic_div[lang][t][k] = jsd
		for k in range(self.K):
			for t in range(self.T-1):
				for lang in self.languages:
					topic1 = softmax(self.phi[lang][t][k])
					topic2 = softmax(self.phi[lang][t+1][k])
					jsd = self.compute_jsd(topic1, topic2)
					time_div[lang][k][t] = jsd
		for lang in self.languages:
			topic_mean = np.mean([np.mean(topic) for topic in topic_div[lang]])
			time_mean = np.mean([np.mean(timet) for timet in time_div[lang]])
			print("Topic JSD -", lang, ":", topic_mean)
			print("Time JSD -", lang, ":", time_mean, "\n")

	def calculate_counts(self):
		m = {lang: [] for lang in self.languages}
		n = {lang: [] for lang in self.languages}
		n_sum = {lang: [] for lang in self.languages}
		for t in range(self.T):
			for lang in self.languages:
				# get the counts for time slice t
				m_t = np.array([[0.0 for topic in range(self.K)] for doc in range(self.timeslices[t])])
				n_t = np.array([[0.0 for word in range(self.V[lang])] for topic in range(self.K)])
				n_sum_t = np.array([0.0 for k in range(self.K)])
				for d in range(self.timeslices[t]):
					for w in range(self.N[lang][t][d]):
						topic = self.z[lang][t][d][w]
						word_id = self.w[lang][t][d][w]
						m_t[d][topic] += 1.0
						n_t[topic][word_id] += 1.0
						n_sum_t[topic] += 1.0
				m[lang].append(m_t)
				n[lang].append(n_t)
				n_sum[lang].append(n_sum_t)
		return m, n, n_sum

	def compute_theta_phi(self):
		theta = []
		phi = {lang: np.empty(shape=(self.T, self.K, self.V[lang]), dtype=float) for lang in self.languages}
		for t in range(self.T):
			theta_t = np.array([[0.0 for topic in range(self.K)] for doc in range(self.timeslices[t])])
			for d in range(self.timeslices[t]):
				for lang in self.languages:
					theta_t[d] = np.add(theta_t[d], self.counts['m'][lang][t][d])
				theta_t[d] = np.array([theta_t[d]/np.sum(theta_t[d])])
			theta.append(theta_t)
			for lang in self.languages:
				phi_t = np.copy(self.counts['n'][lang][t])
				for k in range(self.K):
					if np.sum(phi_t[k]) == 0:
						phi_t[k] = np.asarray([1.0/len(phi_t[k]) for _ in range(len(phi_t[k]))])
					else:
						phi_t[k] = 1.0*phi_t[k]/np.sum(phi_t[k])
				phi[lang][t] = phi_t
		return theta, phi

	def compute_theta(self):
		theta = []
		for t in range(self.T):
			theta_t = np.array([[0.0 for topic in range(self.K)] for doc in range(self.timeslices[t])])
			for d in range(self.timeslices[t]):
				for lang in self.languages:
					theta_t[d] = np.add(theta_t[d], self.counts['m'][lang][t][d])
				theta_t[d] = np.array([theta_t[d]/np.sum(theta_t[d])])
			theta.append(theta_t)
		return theta

	def compute_phi(self):
		phi = {lang: np.empty(shape=(self.T, self.K, self.V[lang]), dtype=float) for lang in self.languages}
		for t in range(self.T):
			for lang in self.languages:
				phi_t = np.copy(self.counts['n'][lang][t])
				for k in range(self.K):
					if np.sum(phi_t[k]) == 0:
						phi_t[k] = np.asarray([1.0/len(phi_t[k]) for _ in range(len(phi_t[k]))])
					else:
						phi_t[k] = 1.0*phi_t[k]/np.sum(phi_t[k])
				phi[lang][t] = phi_t
		return phi

	def get_learning_rate(self, i):
		lr = 0.5 * (90+i)**-0.70
		return lr

	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def get_noise(self, lr):
		return np.random.normal(0, lr)

	def sample_z(self, ts, doc_index, word_id, lang):
		theta = self.theta[ts][doc_index]
		phi = self.phi[lang][ts]
		topic_prob = [np.exp(theta[k]) * np.exp(phi[k][word_id]) for k in range(self.K)]
		topic_prob = topic_prob/np.sum(topic_prob)
		new_topic = list(np.random.multinomial(1, topic_prob, size=1)[0]).index(1)
		return new_topic

	def sample_phi(self, k, iteration):
		lr = self.get_learning_rate(iteration)
		noise = self.get_noise(lr)
		for lang in self.languages:
			phi = self.phi[lang]
			for ts in range(self.T):
				prob_phi = self.softmax(phi[ts][k])
				for w in range(self.V[lang]):
					if ts == 0:
						left = phi[ts+1][k][w] - phi[ts][k][w]
					elif ts == self.T-1:
						left = phi[ts-1][k][w] - phi[ts][k][w]
					else:
						left = (phi[ts+1][k][w] + phi[ts-1][k][w]) - 2*phi[ts][k][w]
					left /= self.beta**2
					right = self.counts['n'][lang][ts][k][w] - (self.counts['n_sum'][lang][ts][k] * prob_phi[w])
					gradient_w = left + right
					delta_phi = (0.5*lr) * gradient_w + noise
					phi[ts][k][w] += delta_phi

	def sample_theta(self, ts, doc_index, iter):
		theta = self.theta[ts][doc_index]
		prob_theta = self.softmax(theta)
		lr = self.get_learning_rate(iter)
		noise = self.get_noise(lr)
		alpha = self.alpha[ts]
		for k in range(self.K):
			left = -1/self.psi**2 * (theta[k] - alpha[k])
			right = 0
			for lang in self.languages:
				n_dt = self.N[lang][ts][doc_index]
				right += (self.counts['m'][lang][ts][doc_index][k] - (n_dt * prob_theta[k]))
			gradient_theta = left + right
			delta_theta = (0.5*lr) * gradient_theta + noise
			theta[k] += delta_theta

	def sample_alpha(self):
		alpha = self.alpha
		for ts in range(1, self.T):
			if ts == 0:
				alpha_mean = alpha[ts+1]
			elif ts == self.T-1:
				alpha_mean = alpha[ts-1]
			else:
				alpha_mean = (alpha[ts-1] + alpha[ts+1]) / 2
			theta = self.theta[ts]
			theta_mean = np.mean(theta, axis=0)
			dt = self.timeslices[ts]
			identity_mat = np.identity(self.K, dtype=float)
			cov_hat = (2/self.sigma**2 + dt/self.psi**2) * identity_mat
			cov_hat_inv = np.linalg.inv(cov_hat)
			minus_term = ((2/self.sigma**2)*theta_mean + (dt/self.psi**2)*alpha_mean)
			minus_term = cov_hat_inv * minus_term
			mu_hat = (alpha_mean + theta_mean) - minus_term.diagonal()
			alpha[ts] = np.random.multivariate_normal(mu_hat, cov_hat_inv)

	def resample_topic(self, lang, ts, doc_id, word):
		word_id = self.w[lang][ts][doc_id][word]
		old_topic = self.z[lang][ts][doc_id][word]
		self.counts['m'][lang][ts][doc_id][old_topic] -= 1
		self.counts['n'][lang][ts][old_topic][word_id] -= 1
		self.counts['n_sum'][lang][ts][old_topic] -= 1
		new_topic = self.sample_z(ts, doc_id, word_id, lang)
		self.z[lang][ts][doc_id][word] = new_topic
		self.counts['m'][lang][ts][doc_id][new_topic] += 1
		self.counts['n'][lang][ts][new_topic][word_id] += 1
		self.counts['n_sum'][lang][ts][new_topic] += 1

	def resample_doc(self, ts, iter):
		print("resample docs in time slice", ts)
		for doc_id in range(self.timeslices[ts]):
			self.sample_theta(ts, doc_id, iter)
			for lang in self.languages:
				for w in range(self.N[lang][ts][doc_id]):
					self.resample_topic(lang, ts, doc_id, w)

	def gibbs_sampling(self):
		time_start = time()
		for it in range(self.n_iterations):
			print("\n--- iteration", str(it+1), "of", self.n_iterations, "---")
			self.sample_alpha()
			thread_list = []
			for k in range(self.K):
				th = Thread(target=self.sample_phi, args=(k, iter, ))
				th.start()
				thread_list.append(th)
			for th in thread_list:
				th.join()
			self.evaluate_divergence()
			for ts in range(self.T):
				self.resample_doc(ts, iter)
		self.theta = self.compute_theta()
		self.phi = self.compute_phi()
		time_duration = (time() - time_start) / (60*60)
		print("Done!")
		print("*** Sampling took ", str(time_duration), " hours ***")
