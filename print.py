import json
import spacy
import csv
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer

import final
import baseModel
import pickle

f = "training.json"
nlp = spacy.load('en')

fp1 = "features.json"
fp2 = "features.csv"

def printJson(filepath):

	stemmer = SnowballStemmer("english")

	sims = []
	total = 0
	bad = 0

	with open(filepath, "r") as fileHandle:
		test_set = json.load(fileHandle)

	for title in test_set["data"]:
		for paragraph in title["paragraphs"]:
			#p = paragraph["context"]
			#print(paragraph["context"])
			sents = sent_tokenize(paragraph["context"])

			for question in paragraph["qas"]:
				total += 1
				q = question["question"]
				
				simScore, sent = final.calSim(q, sents)
				atScore = final.matchAT(final.extractAT(q), nlp(sent))

	fileHandle.close()

	return sims


def df2csv(fp):

	with open(fp, 'rb') as f:
		df = pickle.load(f)
	f.close()

	df.to_csv(r'answers.csv', index = None, header=True)


def json2csv(fp1, fp2):

	with open(fp1, "r") as fileHandle:
		feature_set = json.load(fileHandle)
	fileHandle.close()

	features = feature_set["TrainingFeatures"]

	with open(fp2, "w") as f:
		writer = csv.writer(f)
		writer.writerow(['SimScore', 'ATavail', 'Class'])

		for feature in features:
			golden = feature[1]
			fs = []
			for k, v in feature[0].items():
				fs.append(k)

			writer.writerow([fs[0], fs[1], golden])
	f.close()


def check(fp):

	with open(fp, "r") as fileHandle:
		feature_set = json.load(fileHandle)
	fileHandle.close()

	counterT = 0
	counterF = 0

	features = feature_set["TrainingFeatures"]

	for feature in features:
		if feature[1]:
			counterT += 1
		else:
			counterF += 1

	return counterT, counterF



def queryExtraction(question):
	"""
	Function that extracts and returns the query of a question string [question].
	"""

	#Removing punctuations and making all tokens in lowercase
	tokenize = RegexpTokenizer(r'\w+')
	transTable =  dict((ord(char), None) for char in string.punctuation)
	tokens = question.translate(transTable).lower().split()

	#Removing the question words
	query = [token for token in tokens if token not in QWS]

	return query