import sys
import json
import spacy
from nltk.tokenize import sent_tokenize
from nltk import MaxentClassifier
import final
import baseModel

nlp = spacy.load('en')

def main(fpTrain, fpTest, toJson):

	# fpTrain = "training.json"
	# fpTest = "development.json"

	print("Hello user!")
	print("Training starts...")

	#Training Phase
	classifier = final.trainMaxEnt(fpTrain)

	print("Classifier Trained! Starts predicting...")

	#Prediction Phase
	predict = {}

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


	print("Prediction done! Writing into csv/json file...")

	if toJson:
		baseModel.dict2json(predict, "prediction.json")
	else:
		baseModel.dict2csv(predict, "prediction.csv")



if __name__ == "__main__":

	f = sys.argv[1]
	fp_train = sys.argv[2]
	fp_test = sys.argv[3]

	main(fp_train, fp_test, f == 'json')