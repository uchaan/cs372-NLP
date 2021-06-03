import nltk
from nltk import FreqDist
from nltk.corpus import * 
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# empty list to append output pairs
outputs = [] 
count = 0

# object for stem,lemma check
ls = LancasterStemmer()
lemma = WordNetLemmatizer()

# Filters for POS check
verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjective = ['JJ', 'JJR', 'JJS']
adverb = ['RB']

# Filters for unexpected adverbs 
bad_adverb = ['not','never','so','only','ever','again','once','soon','more','yet','probably','then','away','enough','early','also','even','now','thus']


# Loop in gutenberg corpus 
for fileid in gutenberg.fileids(): 
    # get fileid.txt 
    text = nltk.Text(gutenberg.words(fileid)) 

    # list of different words in text 
    word_list = sorted(set(text))
    # FreqDist of text
    fdist = FreqDist(text) 

    # Loop every words in word_list 
    for word in word_list: 

        word = word.lower() 

        # Get POS of word. Check whether it's verb or adjective. 
        # Else, Skip this word. 
        pos = nltk.pos_tag([word])[0][1] 
        is_verb = 0

        if pos in verb:
            is_verb = 1
            pass 
        elif pos in adjective: 
            pass
        else:
            continue

        # Get synonym set of word using wordnet
        synset = wordnet.synsets(word)

        # If synonym set empty, skip this word
        if len(synset)==0: 
            continue

        # Get First synonym 
        syn_word = synset[0].lemmas()[0].name().lower()
        
        # If word and synonym are same, skip this word
        if syn_word == word:
            continue

        # Or If they are from same Stem or Lemma, skip this word.
        # 1. Stem check by LancasterStemmer
        if ls.stem(word) == ls.stem(syn_word):
            continue
        
        # 2. Lemma check by WordNetLemmatizer
        if is_verb:
            if lemma.lemmatize(word, pos='v') == lemma.lemmatize(syn_word, pos='v'):
                continue
        else:
            if lemma.lemmatize(word, pos='a') == lemma.lemmatize(syn_word, pos='a'):
                continue
    
        # If synonym is also in the same text, I will judge that they are good pair. 
        # Now, Let's find a instensity-adverbs in the text
        if syn_word in word_list:
            # Frequent one is the medium-intensity expression
            if fdist[word] > fdist[syn_word]: 
                i = text.index(word)
                if nltk.pos_tag([text[i-1]])[0][1] in adverb: 
                    if text[i-1].lower() in bad_adverb: # Filter bad adverb
                        outputs.append((word, syn_word))
                        pass
                    else: 
                        outputs.append((text[i-1].lower()+' '+word, syn_word))
                    continue
                elif nltk.pos_tag([text[i+1]])[0][1] in adverb:
                    if text[i+1] in bad_adverb: # Filter bad adverb
                        outputs.append((word, syn_word))
                        pass
                    else:
                        outputs.append((word+' '+text[i+1],syn_word))
                    continue

            elif fdist[word] == fdist[syn_word]: 
                outputs.append((word, syn_word))
                continue

            else: 
                i = text.index(syn_word)
                if nltk.pos_tag([text[i-1]])[0][1] in adverb: 
                    if text[i-1].lower() in bad_adverb:
                        outputs.append((syn_word, word))
                        pass
                    else: 
                        outputs.append((text[i-1].lower()+' '+syn_word, word))
                    continue
                elif nltk.pos_tag([text[i+1]])[0][1] in adverb:
                    if text[i+1] in bad_adverb:
                        outputs.append((syn_word, word))
                        pass
                    else:
                        outputs.append((syn_word+' '+text[i+1], word))
                    continue

    # Let's just try for 1 corpora 
    # break
    count+=1
    if count == 1:
        break

# ===========================
# === Outputs Filtering =====
# ===========================

# 1. filter repitions
outputs = sorted(set(outputs))

# 2. filter stems  (ex: (remember, recollected) = (remember, recollecting) )
already = set()
filtered_outputs = [] 

for output in outputs:
    if output[0] in already:
        continue
    filtered_outputs.append(output)
    already.add(output[0])

print(filtered_outputs)
print('\n\n')
print(len(filtered_outputs))
print('\n\n')






        

