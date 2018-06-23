from textblob.classifiers import NaiveBayesClassifier
import csv

training = []

with open("training_data.tsv","r") as training_data:
    reader = csv.reader(training_data, delimiter='\t')
    for row in reader:
        training.append((row[0],row[1]))

classifier = NaiveBayesClassifier(training)

with open("eval_data.txt") as fil:
    for line in fil:
        classifier.classify(line)
