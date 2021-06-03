import nltk
from nltk import FreqDist
from nltk.corpus import * 
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup 
import ssl
import csv

# =======================================================================
# ==================== Function definition ==============================
# =======================================================================

# Function 'corpus_looper' find proper pairs over input corpus. 
def corpus_looper ( corpus ): 
    for fileid in corpus.fileids(): 
        # get fileid.txt 
        text = nltk.Text(corpus.words(fileid))

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
                        if text[i-1].lower() in intensity_adverbs: # Filter bad adverb
                            outputs_with_intensity.append((text[i-1].lower()+' '+word, syn_word))
                        else: # adverb is inppropriate
                            if len(outputs_without_intensity)<80:
                                outputs_without_intensity.append((word, syn_word))
                            pass
                        continue

                elif fdist[word] == fdist[syn_word]: 
                    # frequency in corpora is same
                    if len(outputs_without_intensity)<80:
                        outputs_without_intensity.append((word, syn_word))
                    continue

                else: 
                    i = text.index(syn_word)
                    if nltk.pos_tag([text[i-1]])[0][1] in adverb: 
                        if text[i-1].lower() in intensity_adverbs:
                            outputs_with_intensity.append((text[i-1].lower()+' '+syn_word, word))
                        else:
                            if len(outputs_without_intensity)<80: 
                                outputs_without_intensity.append((syn_word, word))
                            pass
                        continue

# Function output_filtering filters output (remove overused-adverbs & inappropriate pairs)
def output_filtering (outputs, intensity=True): 
    already = set() 
    adverbs_freq = {}
    filtered_outputs = []
    for output in outputs:
        if (output[0] in already) or (output[1] in already):
            continue

        if intensity: 
            adv = output[0].split()[0]
            if adv in adverbs_freq:
                if adverbs_freq[adv] is 4: 
                    continue
                adverbs_freq[adv]+=1
            else: 
                adverbs_freq[adv] = 1 

        filtered_outputs.append(output)
        already.add(output[0])
        already.add(output[1])

    return filtered_outputs

# =======================================================================
# =======================================================================
# Main process to get outputs. 

# WEBCRAWLING, get Intensity Adverbs examples list from Web
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'https://www.teachingbanyan.com/grammar/adverb-of-degree-intensity-quantity/'
html = urllib.request.urlopen(url,context=ctx).read()
soup = BeautifulSoup(html, 'html.parser')

tags = soup('p')
for tag in tags: 
    if tag.get_text().startswith('almost'):
            intensity_adverbs = tag.get_text().split(', ') 
            # intensity_adverbs = ['almost', 'absolutely', 'awfully', 'badly', 'barely', .... ]

# Filters for unwanted adverbs in intensity_adverbs.
unwanted_adverb = ['how', 'just', 'so', 'guess', 'quite']
intensity_adverbs = list(set(intensity_adverbs) - set(unwanted_adverb))

# empty list to store output pairs
outputs_with_intensity = [] 
outputs_without_intensity = []

# Objects for stem,lemma checking
ls = LancasterStemmer()
lemma = WordNetLemmatizer()

# Filters for POS checking
verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjective = ['JJ', 'JJR', 'JJS']
adverb = ['RB']

# Now, use gutenberg, inaugural, and webtext corpus as inputs of corpus_looper, to get output pairs. 
corpus_looper(gutenberg)
corpus_looper(inaugural)
corpus_looper(webtext)

# === Outputs Filtering =====

# 1. filter repitions
outputs_with_intensity = sorted(set(outputs_with_intensity))
outputs_without_intensity = sorted(set(outputs_without_intensity))

# 2. filter the repeated words from same stem, and over-used adverbs using function output_filtering (ex: (remember, recollected) = (remember, recollecting) )
filtered_outputs_with_intensity = output_filtering(outputs_with_intensity)
filtered_outputs_without_intensity = output_filtering(outputs_without_intensity, intensity=False) 

# Final output is composed with 20 pairs without intensity adverbs, and 30 with intensity adverbs. 
final_output = filtered_outputs_without_intensity[-20:] + filtered_outputs_with_intensity[:30]

print(final_output, '\n', len(final_output))

# Save result to CSV file. 
f = open('CS372_HW1_output_20160255.csv', 'w', encoding='utf-8')
wr = csv.writer(f) 
for (i, pair) in enumerate(final_output):
    wr.writerow([i+1, pair[0], pair[1]])
f.close()
