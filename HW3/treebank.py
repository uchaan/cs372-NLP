import nltk
from nltk.corpus import brown
import math 
from nltk.tag.sequential import ClassifierBasedPOSTagger 
from nltk.corpus import treebank 
from nltk.tag import StanfordPOSTagger
import os
from nltk.tag import tnt


brown_tagged_sents = brown.tagged_sents(categories = 'news')

split_idx = math.floor(len(brown_tagged_sents)*0.8)

# print(len(brown_tagged_sents))
# print(len(treebank.tagged_sents()))

# training = brown_tagged_sents[0:split_idx]
# test = brown_tagged_sents[split_idx:]

# tnt_tagger = tnt.TnT() 
# tnt_tagger.train(brown_tagged_sents)
# print(tnt_tagger.evaluate(treebank.tagged_sents()))

# # treebankTagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')

# # treebankTagger.train(training)

# # print(treebankTagger.evaluate(test))

# print(perceptron_tagger.evaluate(test))

# initializing training and testing set     
# train_data = treebank.tagged_sents()[:3000] 
# test_data = treebank.tagged_sents()[3000:] 
"""
perceptron_tagger = nltk.tag.perceptron.PerceptronTagger(load=False)
perceptron_tagger.train(brown_tagged_sents)

print('Perceptron Tagger accuracy, trained with brown tagged sentences and test with treebank tagged sentences\n', perceptron_tagger.evaluate(treebank.tagged_sents()))

from pickle import dump 
output = open('p_tagger.pkl', 'wb') 
dump(perceptron_tagger, output, -1) 
output.close() 
"""
tagging = ClassifierBasedPOSTagger(train = brown_tagged_sents + treebank.tagged_sents()[:3000]) 
  
a = tagging.evaluate(treebank.tagged_sents()[3000:]) 

from pickle import dump 
output = open('CB_tagger.pkl', 'wb') 
dump(tagging, output, -1) 
output.close() 

print('ClassifierBasedPOSTagger accuracy, trained with brown tagged sentences and test with treebank tagged sentences\n', a) 
