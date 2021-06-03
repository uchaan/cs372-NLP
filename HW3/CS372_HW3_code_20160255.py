import nltk, string, re, pickle, csv
from nltk import FreqDist
from nltk.corpus import * 
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup 
import ssl
import csv

"""
use WordNetLemmatizer for Lemmatizing
"""
lemmatizer = WordNetLemmatizer() 

""" 
use cmudict for heteronym check (>=2 pronouncing)
"""
prondict = cmudict.dict()
prondict = defaultdict(list, prondict)

"""
POS filters
"""
noun = ['NN', 'NNS', 'NNP', 'NNPS']
verb = ['VBD', 'VB', 'VBG', 'VBN', 'VBP']
adjective = ['JJ', 'JJR', 'JJS']

"""
for WebCrawling
"""
american = ['ɚ','ɝ','ɪr','ɛr','ʊr','ɔr','ɑr'] 
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def crawling(word):

    """
    function crawling
    input: word (heteronym) 
    output: given word's pronounciation list, definition list 
    """

    # Referred wiktionary. 
    url = 'https://en.wiktionary.org/wiki/'+word
    html = urllib.request.urlopen(url,context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')

    # Count Etymology numbers
    n = 1 
    while(True): 
        etymology = soup.find("span", {"class": "mw-headline", "id": "Etymology_"+str(n)})
        if etymology==None:
            break
        if len(etymology.get_text().split()) == 1 :
            break
        n+=1
    n-=1

    # Find pronounciations
    pron_list = [] 
    pron = soup.find_all("span", {"class": "IPA"})
    count = 0
    for p in pron: 
        p = p.get_text()
        if p.count('/')==2:
            pron_list.append(p)
            count +=1 

        if count == n:
            break
    
    # Find definitions
    definition_list = [] 
    definition = soup.find_all("ol")
    for i in range(n): 
        definition_list.append(definition[i].get_text().split('\n')[0])

    return (pron_list, definition_list)

def treebankTag(sentence):
    """
    Tree bank tagger, instead of default pos_tagger 
    input: tokenized sentence
    output: tagged tokenized entence
    """
    treebankTagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
    return treebankTagger.tag(sentence)

def has_heteronyms(sentence): 
    """
    function has_heteronyms
    input: single sentence (list)
    output: number of heteronyms in the sentence 
    """
    heteronyms = [] 
    tagged_sentence = treebankTag(sentence)

    for (word, tag) in tagged_sentence:
        # POS Check and Lemmatization 
        if tag in noun+verb+adjective: # Tag is one of N/V/ADJ
            if tag in verb:
                word_lemma = lemmatizer.lemmatize(word, pos='v')
            elif tag in adjective:
                word_lemma = lemmatizer.lemmatize(word, pos='a')
            else: 
                word_lemma = lemmatizer.lemmatize(word)
            
            #pronounce check
            if len(prondict[word_lemma])>=2: # word has >=2 pronouncing
    
                #meaning check
                synset = wn.synsets(word_lemma) 
                if len(synset)>=2: 
                    for syn in synset: 
                        if syn.name().split('.')[0]==word_lemma :
                            sim = synset[0].wup_similarity(syn)
                            # similarity 0.13 is threshold for heteronym 
                            if sim is not None and sim < 0.13: 
                                heteronyms.append(word_lemma)
                                break
    
    # heteronyms: [(heternonym, position in sentence)]
    return heteronyms

def sentence_similarity(definition, sentence):
    """
    calculate the similarity btwn sentence, and definition
    input: tokenized definition (list), tokenized sentence(list)
    output: similarity score 
    """
    total = 0
    tagged_definition = treebankTag(definition)
    tagged_sentence = treebankTag(sentence)

    # Sum the similarity between all words in definition and sentence
    for (word1,tag1) in tagged_definition:
        for (word2,tag2) in tagged_sentence:
            target_pos = adjective + noun + verb
            if tag1 in target_pos and tag2 in target_pos:
                if tag1 in verb:
                    word1 = lemmatizer.lemmatize(word1, pos='v')
                elif tag1 in adjective:
                    word1 = lemmatizer.lemmatize(word1, pos='a')
                else: 
                    word1 = lemmatizer.lemmatize(word1)

                if tag2 in verb:
                    word2 = lemmatizer.lemmatize(word2, pos='v')
                elif tag2 in adjective:
                    word2 = lemmatizer.lemmatize(word2, pos='a')
                else: 
                    word2 = lemmatizer.lemmatize(word2)

                w1 = wn.synsets(word1)
                w2 = wn.synsets(word2)

                # Similarity calculation
                if len(w1)==0 or len(w2)==0: 
                    pass 
                else: 
                    point = w1[0].wup_similarity(w2[0])
                    if point is None: 
                        total+=0
                    else: 
                        total +=point 
    return total

def find_annotation(heteronym_sentence):
    """
    Find the annotation of heteronyms in the sentence

    input: heternoym_sentence : list of tuples (n, heteronyms, sentence)
    output: annotation added to the input, list of tuples (n, heteronyms, sentence, pronounce)

    """
    n, heteronyms, sentence = heteronym_sentence
    pronounce = []
    for heteronym in heteronyms:
        pron_list, definition_list = crawling(heteronym)
        # definition, pronounce number not matching
        score = []
        if len(pron_list)!=len(definition_list) or len(pron_list)==0 or len(definition_list)==0:
            n-=1
            if n==1:
                #fail
                return ()
        else:
            for (i,definition) in enumerate(definition_list):
                definition = nltk.word_tokenize(definition)
                sim = sentence_similarity(definition, sentence)
                score.append(sim)
        
        if len(score) == 0:
            return () 
        pronounce.append(pron_list[score.index(max(score))])

    return (n, heteronyms, sentence, pronounce)

def score_for_ranking(heteronym_sentence_annotated): 

    """
    score the heternoym sentences with the criteria of hw3 description
    input: annotated heteronym sentence
    output: score + annotated heteronym sentence

    """

    n, heteronyms, sentence, pronouce = heteronym_sentence_annotated
    score = 0 
    # 1 occurences of homographs (+3)
    score += n * 3 

    # 2 multiple occurences of homographs. (+2)
    multiple_occurence = defaultdict(int) 
    for heteronym in heteronyms: 
        multiple_occurence[heteronym] += 1
    
    for i in multiple_occurence.values() :
        score += (i-1) * 2

    # 3 if repeated, same POS ? (+1)
    tagged_sentence = treebankTag(sentence)
    heteronyms_tag = [] 
    for (word, tag) in tagged_sentence:
        if tag in noun+verb+adjective: # Tag is one of N/V/ADJ
            if tag in verb:
                word_lemma = lemmatizer.lemmatize(word, pos='v')
            elif tag in adjective:
                word_lemma = lemmatizer.lemmatize(word, pos='a')
            else: 
                word_lemma = lemmatizer.lemmatize(word)
            
            if word_lemma in heteronyms:
                heteronyms_tag.append((word_lemma, tag))

    same_tag = defaultdict(int) 
    for heteronym_tag in heteronyms_tag:
        same_tag[heteronym_tag] += 1
    
    for i in same_tag.values():
        score += (i-1) 

    heteronym_sentence_annotated_scored = (score, n, heteronyms, sentence, pronouce)
    return heteronym_sentence_annotated_scored


# Main function of this code 

heteronym_sentences = [] 
"""
heteronym_sentences:
(number of heteronyms, heteronyms in the sentence, sentence) 
"""

try: 
    # Find saved data
    with open('heteronym_sentences.txt', 'rb') as f:
        heteronym_sentences = pickle.load(f) 
    
except FileNotFoundError:
    # No saved data
    print("No saved file\n")


    # Find sentences that contains 2 or more heteronyms, in brown corpus. 
    for fileid in brown.fileids():
        sentences = brown.sents(fileid)
        for sentence in sentences: 
            heteronyms = has_heteronyms(sentence)
            # heteronyms = [(heternoym, position) , ..] 
            if len(heteronyms)>=2: 
                n = len(heteronyms)
                print((n, heteronyms))
                heteronym_sentences.append((n, heteronyms, sentence))
                pass
    
    # Save the sentences. 
    with open('heteronym_sentences.txt', 'wb') as f: 
        pickle.dump(heteronym_sentences, f)



heteronym_sentences_annotated =[]
"""
heteronym_sentences_annotated:
(number of heteronyms, heteronyms in the sentence, sentence, pronounce) 
"""

try: 
    # Find saved data
    with open('heteronym_sentences_annotated.txt', 'rb') as f:
        heteronym_sentences_annotated = pickle.load(f) 
except FileNotFoundError:

    # Find annotation of heteronyms in sentence
    for heteronym_sentence in heteronym_sentences:
        annoted = find_annotation(heteronym_sentence)
        if len(annoted) != 0: 
            heteronym_sentences_annotated.append(annoted)
            # print(annoted)

    # Save the annotated sentences. 
    with open('heteronym_sentences_annotated.txt', 'wb') as f: 
        pickle.dump(heteronym_sentences_annotated, f)


heteronym_sentences_annotated_scored = [] 
"""
heteronym_sentences_annotated_scored:
(score, number of heteronyms, heteronyms in the sentence, sentence, pronounce) 
"""

# Calculate the scores. 
for heteronym_sentence_annotated in heteronym_sentences_annotated:
    scored = score_for_ranking(heteronym_sentence_annotated)
    heteronym_sentences_annotated_scored.append(scored)

# Filter to final output based on score. 
duplicate = defaultdict(int) 
final_output = [] 
"""
final output: 
(score, sentence, heteronyms - pronounce, source)
"""
for (score, n, heteronyms, sentence, pronounce) in heteronym_sentences_annotated_scored:
    if duplicate[tuple(heteronyms)]==3: 
        continue

    # put to final output 
    duplicate[tuple(heteronyms)]+=1

    sentence = ' '.join(sentence)
    for i in range(len(heteronyms)):
        heteronyms[i] = heteronyms[i] + ' (' + pronounce[i] + ')'
    
    final_output.append((score, sentence, heteronyms, "brown corpus"))

final_output = sorted(final_output, reverse = True)[:30]


# Save final output into .CSV file. 
f = open('CS372_HW3_output_20160255.csv', 'w', encoding='utf-8')
wr = csv.writer(f) 
for (i, pair) in enumerate(final_output):
    wr.writerow([i+1, pair[0], pair[1], pair[2], pair[3]])
f.close()
