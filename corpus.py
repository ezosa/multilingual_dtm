import os
import numpy as np
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from datetime import date
import calendar
import tarfile
import random

exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stopwords_yle = set(stopwords.words('finnish') +
			['saada','tehdä','jo','sanoa','voi','tulla','muun','myös','jälkeen','pitää','vuosi','mennä','vielä','000']).union(stopwords.words('swedish') +
			['också','få','vilja','ho','säga','fyra','gå','få','in','göra','år','komma','måste','hava','http','föra','taga','enligt',
			 'kunna','bliva','mången','böra','andraga','fjol','mycken','del','000'])

stopwords_denews = set(stopwords.words('english') +
				['according','mr','said','could','would','today','should','shall']).union(stopwords.words('german') +
				['dass', 'fuer', 'sei', 'ueber', 'sagte','sollen','wollen','heute','seien','wuerden','mehr'])

def clean_yle_doc(doc):
    clean_short = " ".join([tok for tok in doc if len(tok) > 2])
    clean_punc = ''.join(ch for ch in clean_short if ch not in exclude)
    clean_stop = " ".join([i for i in clean_punc.lower().split() if i not in stopwords_yle])
    return clean_stop

def clean_denews_doc(doc):
	clean_xml = " ".join([line for line in doc if line[0] != "<"])
	clean_punc = ''.join(ch for ch in clean_xml if ch not in exclude)
	clean_stop = " ".join([i for i in clean_punc.lower().split() if i not in stopwords_denews and len(i) > 2])
	clean_doc = " ".join(lemma.lemmatize(word) for word in clean_stop.split())
	clean_doc = " ".join(word for word in clean_doc.split())
	return clean_doc

def getKey(item):
    return item[1]

def compute_frequency_scores(documents):
    languages = list(documents.keys())
    scores = {}
    for lang in languages:
        articles = [d for docs in documents[lang] for d in docs]
        tokens = [token for art in articles for token in art]
        counts = Counter(tokens)
        tuples = [(key, counts[key]) for key in counts.keys()]
        sorted_tuples = sorted(tuples, key=getKey, reverse=True)
        scores[lang] = sorted_tuples
    return scores

def prune_vocabulary(documents, vocab_len=2000):
    term_scores = compute_frequency_scores(documents)
    languages = list(documents.keys())
    time_slices = len(documents[languages[0]])
    dictionary = {lang: set() for lang in languages}
    for lang in languages:
        valid_tokens = [term[0] for term in term_scores[lang][:vocab_len]]
        for t in range(time_slices):
            n_docs = len(documents[lang][t])
            for d in range(n_docs):
                doc = documents[lang][t][d]
                pruned_doc = [w for w in doc if w in valid_tokens and len(w) > 2]
                documents[lang][t][d] = pruned_doc
                dictionary[lang].update(pruned_doc)
    for lang in languages:
        dictionary[lang] = list(dictionary[lang])
    return documents, dictionary

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)

def get_yle_corpus(n_timeslices):
    print("Getting YLE corpus for", n_timeslices,"time slices")
    yle_filepath = "/wrk/users/zosa/codes/pimlico_store/yle_preprocess3/main/lemmatize/lemmas/data/"
    print("Reading lemmatized articles from ", yle_filepath)
    articles = {}
    tar_files = os.listdir(yle_filepath)
    for tar_file in tar_files:
        tar = tarfile.open(yle_filepath + "/" + tar_file, "r")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                filename = member.name
                print("Filename: ", filename)
                text = f.read().decode('utf-8')
                lines = text.split("|DatePublished ")
                for art in lines:
                    if len(art) > 0:
                        a = art.split("|")
                        date_pub = a[0]
                        art_no = a[1].split()[1]
                        text = a[2]
                        lang = "fi" if "fi" in filename else "sv"
                        if art_no not in articles.keys():
                            articles[art_no] = {}
                            d = date_pub.split("-")
                            articles[art_no]['date'] = d[0] + d[1]
                        articles[art_no][lang] = text
    start_date = date(year=2012, month=1, day=1)
    end_date = add_months(start_date, n_timeslices - 1)
    if end_date.month < 10:
        end_date_str = str(end_date.year) + "0" + str(end_date.month)
    else:
        end_date_str = str(end_date.year) + str(end_date.month)
    end_date_int = int(end_date_str)
    languages = ['fi', 'sv']
    documents = {lang: [] for lang in languages}
    dictionary = {lang: set() for lang in languages}
    timestamps = []
    keys = list(articles.keys())
    for k in keys:
        art = articles[k]
        if int(art['date']) <= end_date_int:
            for lang in languages:
                doc = art[lang]
                clean_doc = clean_yle_doc(doc.split()).split()
                documents[lang].append(clean_doc)
                dictionary[lang].update(clean_doc)
            timestamps.append(art['date'])
    dictionary = {lang: list(dictionary[lang]) for lang in languages}
    unique_timestamps = list(set(timestamps))
    unique_timestamps.sort()
    documents = {lang: np.array(documents[lang]) for lang in languages}
    timestamps = np.array(timestamps)
    documents_sliced = {lang: [] for lang in languages}
    for t in unique_timestamps:
        for lang in languages:
            docs_t = documents[lang][timestamps == t]
            documents_sliced[lang].append(docs_t)
    print("time slices: ", len(unique_timestamps))
    return documents_sliced, unique_timestamps, dictionary


def sample_yle_articles(documents, max_doc):
    print("Sampling", max_doc, "articles for each time slice")
    languages = list(documents.keys())
    lang1 = languages[0]
    documents = {lang: np.array(documents[lang]) for lang in languages}
    documents_sampled = {lang: [] for lang in languages}
    timeslices = len(documents[lang1])
    for t in range(timeslices):
        n_docs = len(documents[lang1][t])
        if n_docs > max_doc:
            random_indexes = random.sample(range(n_docs), max_doc)
            for lang in languages:
                random_docs = documents[lang][t][random_indexes]
                documents_sampled[lang].append(random_docs)
        else:
            for lang in languages:
                documents_sampled[lang].append(documents[lang][t])
    return documents_sampled

def get_denews_corpus(path):
    print("Getting DE-News corpus from: ", path)
	filenames = os.listdir(path)
	filenames.sort()
    languages = ['english', 'german']
	documents = {lang: [] for lang in languages}
	timestamps = {lang:[] for lang in languages}
	dictionary = {lang: set() for lang in languages}
	for f in filenames:
		text = open(path + "/" + f, 'r').read().split()
		index_start = list(np.where(np.array(text) == "<DOC")[0])
		lang = "english" if "en.txt" in f else "german"
		for i in range(len(index_start) - 1):
			start_art = index_start[i] + 2
			end_art = index_start[i + 1]
			article = clean_denews_doc(text[start_art:end_art]).split()
			documents[lang].append(article)
			timestamp = float(f.split("-")[-3]+f.split("-")[-2])
			timestamps[lang].append(timestamp)
			dictionary[lang].update(set(article))
	dictionary = {lang: list(dictionary[lang]) for lang in languages}
	unique_timestamps = list(set(timestamps[languages[0]]))
	unique_timestamps.sort()
	documents = {lang: np.array(documents[lang]) for lang in languages}
	timestamps = {lang: np.array(timestamps[lang]) for lang in languages}
	documents_sliced = {lang: [] for lang in languages}
	for t in unique_timestamps:
		for lang in languages:
			docs_t = documents[lang][timestamps[lang]==t]
			documents_sliced[lang].append(docs_t)
	print("time slices: ", len(unique_timestamps))
	return documents_sliced, unique_timestamps, dictionary
