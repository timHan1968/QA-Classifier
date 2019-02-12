Note*: This is a previous group project imported from the Cornell-hosted github. 

Original link: https://github.coecis.cornell.edu/xh87/NLP-a4


# QA Classifier
 
An sub-routine for a Question Answering system which classfies whether the answer to a given question can be extracted from a given text. 

Implemented two versions for comparison:
1. A Baseline model using hand-written rules
2. A feature-based logistic regression model using ML 

Package Dependency:
1. Numpy
2. Spacy
3. nltk

-----------------------------------------------------------------------------------------------------
**Baseline Model:**

File: [baseModel.py]

Instruction:

Open terminal, navigate to the python file folder, and type the following:

"python baseModel.py json testing.json blPredict.json"

which would predict on [testing.json] and writes the prediction results into a json file named [blPredict.json], 
while doing

"python baseModel.py csv testing.json blPredict.csv"

will do the same thing but output the file to a csv file named [blPredict.csv] instead.

-----------------------------------------------------------------------------------------------------

**Final Model:**

File: [final.py], [main.py]

Instruction:

Open terminal, navigate to the python file folder, and type the following:

"python json training.json testing.json"

would train the classifier on [training.json], do prediction on [testing.json] output the prediction results 
into a json file defaultly named [prediction.json], while doing:

"python csv training.json testing.json"

would instead output the prediction into a csv file defaulty named [prediction.csv].

-----------------------------------------------------------------------------------------------------
Note*: If training on "training.json", it might take about 20 min to finish writing predictions. 
