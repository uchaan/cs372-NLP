import nltk, string, re, pickle, csv
from nltk import FreqDist
from nltk.corpus import * 
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from collections import defaultdict

"""
use WordNetLemmatizer and LancasterStemmer 
"""
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()

def tagset_preprocessing(w_t):  

    """
    function tagset_preprocessing takes a list of tuples (word, tag) as an input, and return preprocessed list of input. 
    Preprocessing: 1) lowercase 2) punctuations removal 3) stopwords removal 
    """

    w_t = [(w.lower(),t) for (w,t) in w_t] # words to lowercase 
    punc = '[' + string.punctuation + ']' 
    w_t = [(w,t) for (w,t) in w_t if not re.search(punc, w)] # remove punctuations 
    w_t = [(w,t) for (w,t) in w_t if w not in stopwords.words('english')] # remove stopwords 

    return w_t 

"""
Filters for POS check, I used universal tagset 
"""
verb = ['VERB']
adjective = ['ADJ']
adverb = ['ADV']

"""
Create FreqDist and Tagset for given corpus (brown) 
"""
word_fdist = FreqDist() # FreqDist of words in corpus
bigram_fdist = FreqDist() # FreqDist of bigrams in corpus
word_tag = [] # (word,POS) list

"""
1. Preparing data
It takes a while to build complete FreqDist and Tagset of brown corpus. 
So I use Pickle to save the data.
"""
try: 
    # If saved file exist, load it
    with open('word_fdist.txt', 'rb') as f1:
        word_fdist = pickle.load(f1)

    with open('bigram_fdist.txt', 'rb') as f2: 
        bigram_fdist = pickle.load(f2)
    
    with open('word_tag.txt', 'rb') as f3: 
        word_tag = pickle.load(f3)

except FileNotFoundError: 
    print("file does not exist")
    print() 

    """
    Brown corpus (tagged)
    """
    word_tag = brown.tagged_words(tagset='universal') # Load universal tagset
    word_tag = tagset_preprocessing(word_tag) # Preprocess
    words_list = [w for (w,t) in word_tag] # Extract words from tagset
    word_fdist = FreqDist(words_list) # FreqDist of words in brown corpus
    bigram_fdist = FreqDist(nltk.bigrams(words_list)) # FreqDist of bigrams in brown corpus (will use for pairs)

    # Save data
    with open('word_fdist.txt', 'wb') as f1: 
        pickle.dump(word_fdist, f1)
    
    with open('bigram_fdist.txt', 'wb') as f2:
        pickle.dump(bigram_fdist, f2)
    
    with open('word_tag.txt', 'wb') as f3:
        pickle.dump(word_tag, f3)

"""
2. Derive a Constant (C) for score evaluation
"""
adv_count = 0 # number of adverbs in brown corpus
pair_count = 0 # number of adverb + adjective pairs in brown corpus

for (i, item) in enumerate(word_tag): 
    word, tag = item 
    if tag in adverb: # word is an adverb  
        next_word, next_tag = word_tag[i+1] # Let's check next word
        adv_count +=1 
        if next_tag in adjective: # next word is adjectiev!
            pair_count += 1
    
C = adv_count / pair_count  # Constant C is a ratio of adv/ adv+adj pairs in brown corpus
C = round(C,1) 

print(C)
print() 

"""
3. Find (D,E) pairs and evaluate the score of uniqueness
"""
result_list = [] # elements : Tuple (score, D, E) 

for (i, item) in enumerate(word_tag): 
    word, tag = item 
    
    # get lemma and stem of given word 
    lemma = lemmatizer.lemmatize(word) 
    stem = stemmer.stem(word) 

    if tag in adverb: # word is adverb 
        next_word, next_tag = word_tag[i+1] 
        if next_tag in adjective: # next word is adjective 
            # Calculate score;   Freq(D) + Freq(lemma of D) + Freq(stem of D) - C x Freq( D + E) 
            score = word_fdist[word]+ word_fdist[lemma] + word_fdist[stem] - C * bigram_fdist[(word, next_word)]
            # put it to the result list  
            result_list.append((score, word, next_word))

# result processing
result_list = list(set(result_list)) # remove duplicated pairs 
result_list = sorted(result_list, key = lambda x: x[0]) # sort it with score. Lower Score, more Uniqueness 
 
count = 0 # outputs count
score_dic = defaultdict(int) # Limit of number of pairs with same score
already_adverbs = [] # Limit same adverbs (D) 

# Final output
final_result = [] # (ranking, D, E)
for (i, item) in enumerate(result_list): 
    if count == 100 : # only 100 outputs
        break
    score, D, E = item 
    if score_dic[str(score)] == 5 : # Same score limitation: max 5
        continue

    if D in already_adverbs: 
        continue
    print(item ,'\n') 
    final_result.append([count+1, D, E]) # add to final result

    already_adverbs.append(D)  
    score_dic[str(score)]+=1
    count+=1

with open('CS372_HW2_output_20160255.csv', 'w', encoding='utf-8', newline='') as f: 
    wr = csv.writer(f) 
    for item in final_result:
        wr.writerow(item)






    

