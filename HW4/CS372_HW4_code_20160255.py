import nltk, string, re, pickle, csv, random
from nltk import FreqDist
from nltk.corpus import * 
from collections import defaultdict
from nltk.corpus import brown
from nltk.tag.sequential import ClassifierBasedPOSTagger 
from nltk.corpus import treebank 
from nltk.tag import DefaultTagger

# unknown tag filter
unknown_tags = ['-NONE-', '-LRB-', 'LS', 'UH']

# Biomedical tag filter
# Sometimes frequent using biomedical vocabularies are not tagged as Noun. (unexpected tag)
medical_vocab = ['cell', 'T', 'apoptosis', 'autophagy', 'tumor','enzyme', 'stimuli', 'nerve', 'death', 'pathogen', 'protein', \
    'synthesis', 'opioid', 'immune', 'pathogenesis', 'liver', 'amino', 'acid', 'nucleus', 'metabolism', 'caspase', 'mitophagy', \
        'stress', 'kinase', 'pathway', 'syndrome', 'cytokine', 'transmembrane', 'chocolate', \
        'diagnosis', 'cereblon', 'receptor']

# action verbs filter
target_verbs = ['activate', 'activated', 'activates', 'inhibit', 'inhibits', 'inhibited', 'bind', 'binds', 'accelerate', \
    'accelerates', 'accelerated', 'augment','augmented', 'augments', 'induce', 'induced', 'induces', 'stimulate', 'stimulates', \
        'stimulated', 'require', 'requires', 'required', 'up-regulate', 'up-regulated', 'up-regulates', 'abolish', 'abolishes', \
            'abolished', 'block', 'blocked', 'blocks', 'down-regulate', 'down-regulates', 'down-regulated', 'prevent', 'prevents', 'prevented']

def Relation_Extraction(sentence_set, tagger, training):
    """ 
    function Relation_Extraction
    input: 
    1. a list of tuples (sentence, annotation) . Can be training sentences or test sentences. 
    2. Pre-Trained Classifier Based Tagger
    during execution, it will print out the extracted relations. 
    output: A tuple of assessed performance (precision, recall, F-Score)
    """
    number = 1
    TP = 0 # relation detected, and it's correct
    FP = 0 # relation detected, but incorrect
    TN = 0 # relation not detected, and actually not exist
    FN = 0 # relation not detected, but exist

    # grammar for named entitiy recognition (NP Chunk)
    grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
    NP: {<VBG><NP>}
    NP: {<NP><CC><NP>}
    NP: {<NP><VBN><NP>}
    NP: {<NP><IN|RB>+<NP>}
    NP: {<NP><NP>}
    NP: {<NP><VBG>}
    """

    results = [] 

    # Regular Expression Parser
    cp = nltk.RegexpParser(grammar, loop=2)

    # Looping over training sentences. 
    for annotated_sentence in sentence_set: 
        sentence, annotation = annotated_sentence 

        original_sentence = sentence

        # answer : [ X, action, Y ]
        answer = annotation.split(',') 

        # sentence tokenization 
        sentence = nltk.word_tokenize(sentence)

        remove = 0
        position = [0]*len(sentence)
        # remove parentheses words 
        for i, word in enumerate(sentence):
            if word == '(': 
                remove = 1
            if word == ')':
                remove = 0 
                position[i] = 1
            if remove: 
                position[i] = 1
        
        l = len(sentence)
        for i in range(l):
            if position[l-i-1] == 1: 
                del sentence[l-i-1]

        # POS tagging on tokenized sentence 
        tagged_sentence = tagger.tag(sentence)

        # Pre-processing for words, especially biomedical words
        for i, (word, tag) in enumerate(tagged_sentence):
            
            if tag in unknown_tags:
                tagged_sentence[i] = (word, 'NNP')
            
            if '-' in word: 
                tagged_sentence[i] = (word, 'NN')

            if word == 'that':
                tagged_sentence[i] = (word, 'THAT')

            if word.lower() == 'these':
                tagged_sentence[i] = (word, 'THESE')
            
            # proper nouns
            if i != 0 and word[0].isupper():
                tagged_sentence[i] = (word, 'NNP')

            # usually her, his, mine ... doesn't use in abstract
            if tag == 'PRP$':
                tagged_sentence[i] = (word, 'NN')

            # If contains number (ex: PLAC-8, etc.)
            m = re.compile('[0-9]+')
            if m.search(word) == None and tag == 'CD':
                tagged_sentence[i] = (word, 'NN')
            
            if m.search(word) != None: 
                tagged_sentence[i] = (word, 'NN')

            # set the target action verbs. 
            if word in target_verbs: 
                tagged_sentence[i] = (word, 'TARGET') 
            
            upper = 0
            for ch in word: 
                if ch.isupper(): 
                    upper+=1
            if upper >= 2: # VEGFs, RNA, ...
                tagged_sentence[i] = (word, 'NNP')

            # frequent using medical vocab
            if word in medical_vocab:
                tagged_sentence[i] = (word, 'NNP')
            
            if word[:-1] in medical_vocab:
                tagged_sentence[i] = (word, 'NNPS')

        parsed = cp.parse(tagged_sentence)

        result = None

        X_found = 0
        X = ''
        Y = ''
        action = ''

        nearest_NP = None
        second_nearest_NP = None
        
        # Find X(NP), action, and Y(NP). 
        for (i,t) in enumerate(parsed): 
            
            if X_found: 
                if type(t) != tuple:
                    if t.label() == 'NP':
                        for (word, tag) in t.leaves():
                            Y += word + ' '
                        Y = Y[:-1]
                        break

            else:
                if type(t) != tuple:
                    if t.label() == 'NP': 
                        if nearest_NP is not None and second_nearest_NP is None:
                            second_nearest_NP = nearest_NP
                        nearest_NP = t

                else : 
                    if t[1]=='TARGET': 
                        if parsed[i-1][0] != 'and':
                            pass
                        else: 
                            if second_nearest_NP is not None:
                                nearest_NP = second_nearest_NP
                        action = t[0]
                        if nearest_NP == None:
                            X_found = 1 
                            continue
                        for (word, tag) in nearest_NP.leaves():
                            X += word + ' '
                        X = X[:-1]
                        X_found = 1 
        
        result = [X, action, Y]
        # Longer(specific) description is OK,

        if X=='' or Y=='': # Relation not found
            FN += 1

        if (answer[0] in result[0]) and (answer[1] == result[1]) and (answer[2] in result[2]): 
            TP += 1
        else : 
            FP += 1
        
        results.append((original_sentence, answer, result))

        # print(tagged_sentence)
        # print( '**************************' + str(number) + '**************************' )
        # number+=1
        # print() 
        # print('answer: ', answer)
        # print() 
        # print('result: ', result)
        # print() 
        # print(parsed) 

    # saving 
    if training : 
        with open('Middle_CS372_HW4_output_20160255.csv', 'w', encoding = 'utf-8') as f :
            wr = csv.writer(f)
            for (sentence, annotation, result) in results: 
                wr.writerow([sentence, annotation, result])
    
    else : # test 
        with open('Middle_CS372_HW4_output_20160255.csv', 'a', encoding = 'utf-8') as f :
            wr = csv.writer(f)
            for (sentence, annotation, result) in results: 
                wr.writerow([sentence, annotation, None])

    # Performance assessment. 
    Accuracy = (TP + TN) / (FP + FN + TP + TN) 
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_score = 2 * Precision * Recall / (Precision + Recall)

    print('Accuracy: ', Accuracy*100, '%')
    print('Precision: ', Precision*100, '%')
    print('Recall: ', Recall*100, '%')
    print('F_score: ', F_score)

    return (Accuracy, Precision, Recall, F_score)

def training_tagger(): 
    """
    Function training_tagger 
    input: None 
    output: Trained Taggger. 
    """

    # Tagged Sentences for training
    # brown_tagged_sents = brown.tagged_sents()
    treebank_tagged_sents = treebank.tagged_sents() 
    defaultTag = DefaultTagger("NN")
    POStagger = ClassifierBasedPOSTagger(train = treebank_tagged_sents, backoff= defaultTag) 

    return POStagger

""" 
Main Part 
"""
# Load or Store the Tagger. 
try: 
    # Pre-trained tagger exist. 
    with open('tagger.pkl', 'rb') as t:
        POStagger = pickle.load(t) 
    
except FileNotFoundError:
    # Tagger is not trained yet. 

    POStagger = training_tagger()

    with open('tagger.pkl', 'wb') as t: 
        pickle.dump(POStagger, t, -1) 

# Split the data
try: 
    with open('training.txt', 'rb') as train: 
        training_sentences = pickle.load(train)
    with open('test.txt', 'rb') as test:
        test_sentences = pickle.load(test)

except FileNotFoundError: 

    annotated_sentences_set = []

    with open('medline.csv', 'r', encoding='utf-8') as f : 
        rdr = csv.reader(f) 
        for annotated_sentences in rdr : 
            sentence, annotation, _, _, _, _ = annotated_sentences
            annotated_sentences_set.append((sentence, annotation))
        
        random.shuffle(annotated_sentences_set)

        # split 
        training_sentences = annotated_sentences_set[:80]
        test_sentences = annotated_sentences_set[80:]

        # save 
        with open('training.txt', 'wb') as f1: 
            pickle.dump(training_sentences, f1)
        
        with open('test.txt', 'wb') as f2: 
            pickle.dump(test_sentences, f2) 

# Training
print("***Training***\n")
performance_training = Relation_Extraction(training_sentences, POStagger, training = True)

print() 

# Test
print("***Test***\n")
performance_test = Relation_Extraction(test_sentences, POStagger, training= False ) 

# output file 
info = {}
with open('medline.csv', 'r', encoding = 'utf-8') as f :
    rdr = csv.reader(f) 
    for pair in rdr: 
        s, _, title, publisher, year, PMID = pair
        info[s] = (title, publisher, year, PMID)
    
output = []
with open('Middle_CS372_HW4_output_20160255.csv', 'r', encoding = 'utf-8') as f:
    rdr = csv.reader(f) 
    for pair in rdr: 
        s, a, r = pair 
        title, publisher, year, PMID = info[s]
        output.append((s,a,r,title,publisher,year,PMID))

with open('CS372_HW4_output_20160255.csv', 'w', encoding = 'utf-8') as f:
    wr = csv.writer(f) 
    for pair in output: 
        (s,a,r,t,p,y,ID) = pair
        wr.writerow([s,a,r,t,p,y,ID])
