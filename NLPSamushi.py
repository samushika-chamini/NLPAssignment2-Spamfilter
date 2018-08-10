# NLPAssignment2-Spamfilter
NLPAssignment2-Spamfilter B14 IT Faculty - UOM, Sri Lanka
#
# Author : 144190A - W.W.A.S.C.Wickramarathne
# Bundle : NLP Assignment 2
# B14, faculty of Information Technology, University of Moratuwa.
#

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

dataFileValues = pd.read_csv("SMSSpamCollection.tsv", sep='\t', names=['Label', 'Message'])
#change lables into tags
dataFileValues.loc[dataFileValues["Label"] == 'ham', 'Label'] = 1
dataFileValues.loc[dataFileValues["Label"] == 'spam', 'Label'] = 0
messages = dataFileValues['Message']
labels = dataFileValues['Label']

TestCaseEqualData = 0
PredictedArrayLength = 0

vectorClassifier = TfidfVectorizer(min_df=1,stop_words='english', ngram_range=(1,1))
messages_trainData, messages_testData, label_trained, label_testData = train_test_split(messages,labels,test_size=0.1, random_state=4)
x_traincv = vectorClassifier.fit_transform(messages_trainData)
trainData = x_traincv.toarray()
featureNames = vectorClassifier.get_feature_names()
data = vectorClassifier.inverse_transform(trainData[0])
actualValues = messages_trainData.iloc[0]
multinomial = MultinomialNB()
label_trained = label_trained.astype('int')
multiNBData = multinomial.fit(x_traincv, label_trained)
x_testcv = vectorClassifier.transform(messages_testData)
predictionArray = multinomial.predict(x_testcv)
actualTestDataLabels = np.array(label_testData)
testCaseEquality = 0

for i in range (len(label_testData)):
    if (actualTestDataLabels[i] == predictionArray[i]):
        testCaseEquality += 1

TestCaseEqualData = testCaseEquality
PredictedArrayLength = len(predictionArray)

#Uni-gram
print("Unigram Equal testcases count = ",TestCaseEqualData)
print("Unigram Equal testcases count = ",PredictedArrayLength)
print("Unigram Equal testcases count = ", TestCaseEqualData * 100.0 / PredictedArrayLength, "% ~> ", TestCaseEqualData * 100.0 // PredictedArrayLength, "%")



vectorClassifier = TfidfVectorizer(min_df=1,stop_words='english', ngram_range=(2,2))
messages_trainData, messages_testData, label_trained, label_testData = train_test_split(messages,labels,test_size=0.1, random_state=4)
x_traincv = vectorClassifier.fit_transform(messages_trainData)
trainData = x_traincv.toarray()
featureNames = vectorClassifier.get_feature_names()
data = vectorClassifier.inverse_transform(trainData[0])
actualValues = messages_trainData.iloc[0]
multinomial = MultinomialNB()
label_trained = label_trained.astype('int')
multiNBData = multinomial.fit(x_traincv, label_trained)
x_testcv = vectorClassifier.transform(messages_testData)
predictionArray = multinomial.predict(x_testcv)
actualTestDataLabels = np.array(label_testData)
testCaseEquality = 0
for i in range (len(label_testData)):
    if (actualTestDataLabels[i] == predictionArray[i]):
        testCaseEquality += 1
TestCaseEqualData = testCaseEquality
PredictedArrayLength = len(predictionArray)


#Bi-gram
print("Bigram Equal testcases count = ",TestCaseEqualData)
print("Bigram Equal testcases count = ",PredictedArrayLength)
print("Bigram Equal testcases count = ", TestCaseEqualData * 100.0 / PredictedArrayLength, "% ~> ", TestCaseEqualData * 100.0 // PredictedArrayLength, "%")



vectorClassifier = TfidfVectorizer(min_df=1,stop_words='english', ngram_range=(3,3))
messages_trainData, messages_testData, label_trained, label_testData = train_test_split(messages,labels,test_size=0.1, random_state=4)
x_traincv = vectorClassifier.fit_transform(messages_trainData)
trainData = x_traincv.toarray()
featureNames = vectorClassifier.get_feature_names()
data = vectorClassifier.inverse_transform(trainData[0])
actualValues = messages_trainData.iloc[0]
multinomial = MultinomialNB()
label_trained = label_trained.astype('int')
multiNBData = multinomial.fit(x_traincv, label_trained)
x_testcv = vectorClassifier.transform(messages_testData)
predictionArray = multinomial.predict(x_testcv)
actualTestDataLabels = np.array(label_testData)
testCaseEquality = 0
for i in range (len(label_testData)):
    if (actualTestDataLabels[i] == predictionArray[i]):
        testCaseEquality += 1
TestCaseEqualData = testCaseEquality
PredictedArrayLength = len(predictionArray)

#Tri-gram
print("Bigram Equal testcases count = ",TestCaseEqualData)
print("Bigram Equal testcases count = ",PredictedArrayLength)
print("Bigram Equal testcases count = ", TestCaseEqualData * 100.0 / PredictedArrayLength, "% ~> ", TestCaseEqualData * 100.0 // PredictedArrayLength, "%")
