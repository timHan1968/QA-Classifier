import json
import csv
import sys
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


KEYPOS = ['CD', 'JJ', 'JJR', 'JJS', 'LS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR',
'RBS', 'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'FW']
REMOVE = ['do', 'does', 'did', 'is', 'are', 'was', 'were', 'many', 'much']


fp1 = "test.json"
fp2 = "pred.json"


def extractPos(sentence):
	"""
	Function that labels the pos tags on a given sentence.

	Input
	------
	- [sentence]: a string that contains a sentence that could be a question or not

	Output
	------
	- [pos]: a list of tuples in the form [word, POS tag]. 

	"""
	tokens = word_tokenize(sentence)
	pos = pos_tag(tokens)
	return pos



def keywordExt(question):
	"""
	Function that extracts keywords from [question]. The "keyword" is defined as 
	satisfying the following:

	1) The word's POS tag is one of [KEYPOS], which would typically be a noun, verb, 
	   or adjective. 
	2) The word is not one of the common, trivial words as listed in [REMOVE]

	Input
	------
	- [question]: a string that contains a question

	Output
	------
	- [keywords]: a list of strings that are keywords extracted from [question] 

	"""
	stemmer = SnowballStemmer("english")

	posTuples = extractPos(question)
	keywords = [posTuple[0] for posTuple in posTuples if posTuple[1] in KEYPOS 
	and posTuple[0] not in REMOVE]

	keywords = [stemmer.stem(keyword) for keyword in keywords]

	return keywords



def BMpredict(texts, question):
	"""
	Function that returns the baseline model's preidction of whether a [question] 
	is answerable or not using the given [texts].

	Input
	------
	- [texts]: a list of strings (sentences)
	- [question]: a string that contains a question

	Output
	------
	- a boolean value where [True] means answerable and [False] the opposite

	"""
	threshold = 0.50
	stemmer = SnowballStemmer("english")

	keywords = keywordExt(question)

	for text in texts:

		textTokens = word_tokenize(text)
		textTokens = [stemmer.stem(token) for token in textTokens]

		overlap = [keyword for keyword in keywords if keyword in textTokens]

		if len(keywords) == 0:
			#The question is composed by trivial words like "What's that?" which is 
			#configured as unanswerable. 
			continue
		elif len(overlap)/len(keywords) >= threshold:
			return 1 

	return 0



def pred2dict(fp_in):
	"""
	Function that writes the answerable prediction on json files in [fp_in] into a 
	dictionary of format {[Question Id, Prediction]}

	"""
	with open(fp_in, "r") as fileHandle:
		test_set = json.load(fileHandle)

	fileHandle.close()

	dict = {}

	for title in test_set["data"]:
		for paragraph in title["paragraphs"]:
			texts = sent_tokenize(paragraph["context"])
			for question in paragraph["qas"]:
				dict[question["id"]] = BMpredict(texts, question["question"])

	return dict



def dict2csv(dict, fp_out):
	"""
	Function that writes the predictions in [dict] into a csv file at filepath 
	specified by [fp_out]. The input [dict] follows the format {[Question Id, 
	prediction]}.

	"""
	with open(fp_out, "w") as f:
		writer = csv.writer(f)
		writer.writerow(['Id', 'Category'])
		for qid, pred in dict.items():
			writer.writerow([qid, pred])
	f.close()



def dict2json(dict, fp_out):
	"""
	Function that writes the predictions in [dict] into a json file at filepath 
	specified by [fp_out]. The input [dict] follows the format {[Question Id, 
	prediction]}.

	"""
	with open(fp_out, "w") as outputFile:
		json.dump(dict, outputFile)

	outputFile.close()



if __name__ == "__main__":

	fp_in = sys.argv[1]
	fp_out = sys.argv[2]

	dict = pred2dict(fp_in)
	dict2csv(dict, fp_out)

