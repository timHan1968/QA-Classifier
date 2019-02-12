import json
import spacy
from nltk.tokenize import sent_tokenize
from nltk import MaxentClassifier
import final
import baseModel

import pandas as pd
import pickle

nlp = spacy.load('en')


def main():

	fpTrain = "training.json"
	fpTest = "testing.json"
	fpOut = "attempt.csv"
	
	fpFeature = "features.json"

	print("Hello user!")
	print("Training starts...")

	#Training Phase
	classifier = final.trainMaxEnt(fpTrain)

	# with open(fpFeature, "r") as fileHandle:
	# 	feature_set = json.load(fileHandle)
	# fileHandle.close
	# classifier = MaxentClassifier.train(feature_set["TrainingFeatures"], max_iter = 30)

	print("Classifier Trained! Starts predicting...")


	#Prediction Phase
	predict = {}

	counter = 0

	with open(fpTest, "r") as fileHandle:
		test_set = json.load(fileHandle)
	fileHandle.close()

	for title in test_set["data"]:
		for paragraph in title["paragraphs"]:
			sents = sent_tokenize(paragraph["context"])

			for question in paragraph["qas"]:
				q = question["question"]

				simFeature, candSent = final.genSimFeature(q, sents)
				atFeature = final.matchAT(final.extractAT(q), candSent)
				focusFeature = final.genFocusFeature(q, sents)

				features = {simFeature: True, atFeature: True, focusFeature: True}

				predict[question["id"]] = classifier.classify(features)

				counter += 1
				print(counter)


	print("Prediction done! Writing into csv...")

	baseModel.dict2csv(predict, fpOut)


def train(fp):

	with open(fp, "r") as fileHandle:
		feature_set = json.load(fileHandle)
	fileHandle.close

	return MaxentClassifier.train(feature_set["TrainingFeatures"], max_iter = 30)


if __name__ == "__main__":
	main()