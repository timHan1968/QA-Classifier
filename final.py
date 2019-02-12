import json
import spacy
import baseModel
import string
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk import MaxentClassifier

nlp = spacy.load('en')
word2vec = spacy.load('en_core_web_lg')
stemmer = SnowballStemmer("english")


def trainMaxEnt(fp):
	"""
	Function that extracts all features from a training json file at [fp], train a MaxEnt 
	classifier based on these features, and return that classifier

	"""
	with open(fp, "r") as fileHandle:
		test_set = json.load(fileHandle)
	fileHandle.close()

	maxEntCorpus = []

	print("Extracting features from training file...")

	for title in test_set["data"]:
		for paragraph in title["paragraphs"]:
			sents = sent_tokenize(paragraph["context"])

			for question in paragraph["qas"]:
				q = question["question"]

				simFeature, candSent = genSimFeature(q, sents)
				atFeature = matchAT(extractAT(q), candSent)
				focusFeature = genFocusFeature(q, sents)

				features = {simFeature: True, atFeature: True, focusFeature: True}

				if question["is_impossible"]:
					maxEntCorpus.append((features, 0))
				else:
					maxEntCorpus.append((features, 1))

	return MaxentClassifier.train(maxEntCorpus, max_iter = 30)


#Extracting Similarity Feature
#----------------------------------------------------------------
def genSimFeature(question, texts):
	"""
	Function that returns the maximum cosine similarity between [question] and each sentence 
	in [texts] AND the text sentence that has the highest similarity.

	Inputs
	-------
	[question]: the question as a string
	[texts]: a list of strings, each corresponding to a sentence in the text

	"""
	textVecs = [sent2vec(text) for text in texts]
	qVec = sent2vec(question)
	simScores = [cos_sim(qVec, textVec) for textVec in textVecs]
	bestScore = max(simScores)

	return bestScore, texts[simScores.index(bestScore)]


def sent2vec(sentence):
	"""
	Function that returns the word embedding of a given [sentence] by calculating the average 
	of its word embeddings.

	Input
	------
	[sentence]: the sentence as a string

	"""
	tokens = word_tokenize(sentence)
	vectors = [word2vec.vocab[token].vector for token in tokens]

	return (sum(vectors)/len(vectors))


def cos_sim(x, y):
	"""
	The function takes in two vectors [x] and [y] then return their
	cosine similarity.
	"""
	dot_product = np.dot(x, y)
	norm_x = np.linalg.norm(x)
	norm_y = np.linalg.norm(y)

	return dot_product/(norm_x * norm_y)
#----------------------------------------------------------------


#Extracting Focus Feature
#----------------------------------------------------------------
def genFocusFeature(question, sentences):
	"""
	Function that returns highest percentage of question keywords that the 
	[sentences] can have. 

	Inputs
	-------
	[question]: the question as a string
	[sentences]: the setences in the text as a list of strings

	"""
	keywords = baseModel.keywordExt(question)
	stem_sents = [sent2stems(sentence) for sentence in sentences]
	percentages = [findPercent(keywords, stem_sent) for stem_sent in stem_sents]

	return max(percentages)


def findPercent(keywords, sent_stems):
	"""
	Function that returns the percentage of [keywords] present in the list of 
	stem tokens in [sent_stems].

	Input
	------
	[keywords]: the keywords as a list of strings
	[sent_stems]: the stems of tokens in a sentence as a list of strings

	"""
	intersection = [keyword for keyword in keywords if keyword in sent_stems]
	ratio =  len(intersection)/len(keywords)

	return ratio


def sent2stems(sentence):
	"""
	Function that transforms a string of sentence to a list of stemmed tokens.

	Input
	------
	[sentence]: the sentence as a string

	"""
	tokens = word_tokenize(sentence)
	stems = [stemmer.stem(token) for token in tokens]

	return stems
#----------------------------------------------------------------


#Extracting Answer Type Feature
#----------------------------------------------------------------
def matchAT(answerType, sent):
	"""
	Function that returns a pseudo-boolean value to indicate whether 
	the given [sentDoc] contains the desired [answerTypes].

	Input
	------
	[sentDoc]: [nlp] object of a candidate sentence for finding the answer.

	Output
	-------
	0: If the sentence certainly does not contain the desired answered type
	2: If the sentence certainly contains the desired answered type
	1: Whether sentence contains the answer type or not cannot be determined
	"""
	if answerType == []:
		return 1

	sentDoc = nlp(sent)
	NEtags = [x.label_ for x in sentDoc.ents]
	presentTypes = [NEtag for NEtag in NEtags if NEtag in answerType]

	if presentTypes == []:
		return 0
	else:
		return 2



def extractAT(question):
	"""
	A function that extracts the expected answer type of a given question.

	Input
	------
	[question]: the question as a string

	Output
	------
	[answerType]: a list of NER tags which are expected answer types for [question].
	If the question does not have an explicit answer type, the output would just be 
	[].

	"""
	#List of question words
	QWS = ["who", "what", "when", "which", "where", "whi", "how", "whom", "whose"]
	#Match list betweeen question objects and NER type
	OB2AT = {"date": ['DATE'], "time": ['TIME'], "percent": ['PERCENT'], "percentag": ['PERCENT'], 
	"languag": ['LANGUAGE'], "build": ['FAC'], "product": ['PRODUCT'], "name": ['PERSON', 'NORP', 
	'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT','EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'], "compani": ['ORG'], 
	"book": ['WORK_OF_ART'], "song": ['WORK_OF_ART'], "law": ['LAW']}

	tokens = word_tokenize(question)
	stems = [stemmer.stem(token) for token in tokens]
	qWord = [stem for stem in stems if stem in QWS]

	# Note:
	# If [qWord] has a length greater than 1, so multiple question words appear, 
	# we only consider the first one.

	if len(qWord) == 0 and tokens[0] == "Name":
		#Question type: "Name sth sth"
		answerType = OB2AT["name"]

	elif len(qWord) == 0:
		#Questions without explicit question words like "Jack likes hotdogs?"
		answerType = []

	elif qWord[0] in ["who", "whom", "whose"]:
		#Question type 'who'
		answerType = ['PERSON', 'NORP', 'ORG', 'GPE']

	elif qWord[0] == "when":
		#Question type 'when'
		answerType = ['DATE', 'TIME']

	elif qWord[0] == "where":
		#Question type 'where'
		answerType = ['NORP', 'FAC', 'ORG', 'GPE', 'LOC']

	elif qWord[0] =="how":
		#Question type 'how'
		qIdx = stems.index("how")
		if qIdx == len(stems) - 1:
			answerType = []
		else:
			adv = stems[stems.index("how") + 1]
			if adv in ["mani", "much"]:
				answerType = ['PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL']
			else:
				answerType = []

	elif qWord[0] in ["which", "what"]:
		#Question type 'what' or 'which'
		qIdx = stems.index(qWord[0])
		if qIdx == len(stems) - 1:
			answerType = []
		else:
			ob = stems[stems.index(qWord[0]) + 1]
			if ob in OB2AT:
				answerType = OB2AT[ob]
			else:
				answerType = []

	else:
		#Question type "why"
		answerType = []	

	return answerType
#----------------------------------------------------------------