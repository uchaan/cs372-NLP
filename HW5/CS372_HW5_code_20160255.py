# pip install spacy
# python -m spacy download en_core_web_sm

# nltk
import nltk, string, re, pickle, csv, random
from nltk.chunk import ne_chunk
from nltk import FreqDist
from nltk.corpus import * 
from nltk.tokenize import sent_tokenize

from collections import defaultdict

#spaCy
import spacy

#wikipedia api
import wikipediaapi

# count of Feminine and Masculine 
F_pronoun = ['she', 'her', 'herself']
M_pronoun = ['he', 'him', 'himself']

# constants for page-context
beta = 0.5 
gamma = 1.5

# Load English tokenizer, tagger, parser, NER and word vectors of spaCy
nlp = spacy.load("en_core_web_sm")

def load_raw_data():
    """
    function load_raw_data: load the raw data from gap-test.tsv and store in a list
    input: None
    output: gap_raw_data(list)
    """
    with open('gap-test.tsv', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f, delimiter='\t')
        gap_raw_data = list(rdr)
    return gap_raw_data

def Preprocess(gap_raw_data): 
    """
    function Preprocessing: Make a dictionary of data using the ID as a key. 
    input: gap_raw_data (list) 
    output: gap_data_dic (dictionary)
    """

    gap_data_dic = {}
    for line in gap_raw_data: 
        ID, text, pronoun, p_off, A, A_off, A_coref, B, B_off, B_coref, URL = line

        # only use number of ID
        ID = int(ID.split('-')[1])

        # add nlp(text) <spacy object> to the dictionary with other information
        gap_data_dic[ID] = (nlp(text), text, pronoun, p_off, A, A_off, A_coref, B, B_off, B_coref, URL) 

    return gap_data_dic

def Snippet_Context(gap_data_dic):
    """
    function Snippet_Context : operates snippet-context and returns the output result
    input: gap_data_dic(dic)
    output: snippet_output (dictionary of {ID: (TRUE, FALSE), .... })
    """
    snippet_output = {}

    for id in range(1, 2001): # id is 1 - 2000
        doc, text, pronoun, p_off, A, A_off, A_coref, B, B_off, B_coref, _ = gap_data_dic[id]

        # Check whether pronoun, A, B are exist only once in the text. 
        p_count = text.count(pronoun)
        A_count = text.count(A) 
        B_count = text.count(B)

        # tokenized list of text
        tokenized = nltk.word_tokenize(text)

        # find the token index corresponds to the offset
        if p_count>1 or A_count>1 or B_count>1: # repetition, use offset info

            A_index , B_index = -1, -1

            if p_count > 1: # find pronoun token index
                pass

            if A_count > 1: # find A token index
                counted = 0 
                first = 1
                First_word = A.split()[0]
                for i, token in enumerate(tokenized):
                    if first:
                        if First_word == token:
                            if A_coref == 0:
                                A_index = i
                        counted += len(token)
                        first = 0
                    else: 
                        if token in [',', '.', '\'']:
                            counted += 2
                        elif token == First_word:
                            if int(A_off) + 5 > counted and int(A_off) - 5 < counted:
                                A_index = i 
                        else: 
                            counted += 1
                            counted += len(token)
            
            if B_count > 1: # find B token index 
                counted = 0 
                first = 1
                First_word = B.split()[0]
                for i, token in enumerate(tokenized):
                    if first:
                        if First_word == token:
                            if B_coref == 0:
                                B_index = i
                        counted += len(token)
                        first = 0
                    else: 
                        if token in [',', '.', '\'']:
                            counted += 2
                        elif token == First_word:
                            if int(B_off) + 5 > counted and int(B_off) - 5 < counted:
                                B_index = i 
                        else: 
                            counted += 1
                            counted += len(token)

            # find isNP and isNSUBJ
            A_isNP, A_isNSUBJ = 0, 0
            B_isNP, B_isNSUBJ = 0, 0 
            A_isDOBJ, B_isDOBJ = 0, 0
            A_isPOSS, B_isPOSS = 0, 0
            A_len, B_len = len(A.split()), len(B.split())

                
            for chunk in doc.noun_chunks:
                if chunk.text == A :
                    A_isNP = 1
                if chunk.text == B :
                    B_isNP = 1

            if A_count==1:
                for i in range(len(tokenized)):
                    if A.split() == tokenized[i:i+A_len]:
                        A_index = i

            if B_count==1:
                for i in range(len(tokenized)):
                    if B.split() == tokenized[i:i+B_len]:
                        B_index = i

            if A_index is not -1:
                for i in range(A_index, A_index+A_len):
                    if doc[i].dep_ == 'nsubj':
                        A_isNSUBJ = 1
                    if doc[i].dep_ == 'dobj':
                        A_isDOBJ = 1
                    if doc[i].dep_ == 'poss':
                        A_isPOSS = 1
            else: 
                # print("A index not found error in Snippet-Context", A)
                pass

            if B_index is not -1:
                for i in range(B_index, B_index+B_len):
                    if doc[i].dep_ == 'nsubj':
                        B_isNSUBJ = 1
                    if doc[i].dep_ == 'dobj':
                        B_isDOBJ = 1
                    if doc[i].dep_ == 'poss':
                        B_isPOSS = 1
            else:
                # print("B index not found error in Snippet-Context", B)
                pass

            A_pred = False
            B_pred = False

            if A_isNP:
                if A_isNSUBJ: 
                    A_pred = True
                elif A_isPOSS:
                    A_pred = True
            else:
                A_pred = False
            
            if B_isNP:
                if B_isNSUBJ:
                    B_pred = True 
                elif B_isPOSS:
                    B_pred = True
            else: 
                B_pred = False

            if (A_pred, B_pred) == (False, False): 
                if A_isNP and A_isDOBJ:
                    A_pred = True 
                
                if B_isNP and B_isDOBJ:
                    B_pred = True   
                
        else: 
            # NP check
            A_isNP, B_isNP = 0, 0
            for chunk in doc.noun_chunks:
                if chunk.text == A :
                    A_isNP = 1
                elif chunk.text == B:
                    B_isNP = 1
                
            #nsubj check 
            A_isNSUBJ, B_isNSUBJ = 0, 0 
            A_isDOBJ, B_isDOBJ = 0, 0
            A_isPOSS, B_isPOSS = 0, 0
            A_len, B_len = len(A.split()), len(B.split())
            A_index, B_index = -1, -1

            for i in range(len(tokenized)): 
                if A.split() == tokenized[i:i+A_len]:
                    A_index = i
                elif B.split() == tokenized[i:i+B_len]:
                    B_index = i

            if A_index is not -1:
                for i in range(A_index, A_index+A_len):
                    if doc[i].dep_ == 'nsubj':
                        A_isNSUBJ = 1
                    if doc[i].dep_ == 'dobj':
                        A_isDOBJ = 1
                    if doc[i].dep_ == 'poss':
                        A_isPOSS = 1
            else: 
                # print("A index not found error in Snippet-Context", A)
                pass
            
            if B_index is not -1:
                for i in range(B_index, B_index+B_len):
                    if doc[i].dep_ == 'nsubj':
                        B_isNSUBJ = 1
                    if doc[i].dep_ == 'dobj':
                        B_isDOBJ = 1
                    if doc[i].dep_ == 'poss':
                        B_isPOSS = 1
            else:
                # print("B index not found error in Snippet-Context", B)
                pass

            A_pred = False
            B_pred = False

            if A_isNP:
                if A_isNSUBJ: 
                    A_pred = True
                elif A_isPOSS:
                    A_pred = True
            else:
                A_pred = False
            
            if B_isNP:
                if B_isNSUBJ:
                    B_pred = True 
                elif B_isPOSS:
                    B_pred = True
            else: 
                B_pred = False

            if (A_pred, B_pred) == (False, False): 
                if A_isNP and A_isDOBJ:
                    A_pred = True 
                
                if B_isNP and B_isDOBJ:
                    B_pred = True                      

        snippet_output[id] = (A_pred, B_pred)

    return snippet_output

def get_wiki_texts(URL):
    """
    function get_wiki_texts: obtain and return wikipedia texts of given URL using wikipedia api
    input: URL
    output: text of wiki page (String)
    """
    title = URL.split('/')[-1]

    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    p_wiki = wiki_wiki.page(title).text

    return p_wiki

def Page_Context(gap_data_dic, snippet_output): 
    """
    function Page_Context: operates page-context by checking wiki and returns the output result
    input: gap_data_dic(dic), snippet_output(dic)
    output: page_output (dictionary of {ID: (TRUE, FALSE), .... })
    """
    page_output = {}

    for id in range(1, 2001): 
        doc, text, pronoun, p_off, A, A_off, A_coref, B, B_off, B_coref, URL = gap_data_dic[id]

        A_pred_snippet, B_pred_snippet = snippet_output[id]
        
        # get wikipedia text. 
        wiki_text = get_wiki_texts(URL)

        if type(wiki_text) != str:
            print(type(wiki_text), URL)

        # sentence tokenization
        wiki_sentences = sent_tokenize(wiki_text)

        # count name frequency of full-context & near-context 
        # 1. full-context
        A_full = wiki_text.count(A) 
        B_full = wiki_text.count(B)

        # 2. near-context
        A_near, B_near = 0, 0 
        if text in wiki_sentences:
            i = wiki_sentences.index(text)
            if i-1 >=0 and i+1 < len(wiki_sentences):
                A_near = wiki_sentences[i-1].count(A) + wiki_sentences[i+1].count(A)
                B_near = wiki_sentences[i-1].count(B) + wiki_sentences[i+1].count(B)
        
        # 3. context score 
        A_context_score = beta * A_full + gamma * A_near + int(A_pred_snippet)
        B_context_score = beta * B_full + gamma * B_near + int(B_pred_snippet)

        # convert the score to output
        if A_context_score == B_context_score:
            if A_context_score == 0:
                page_output[id] = (False, False)
            else: 
                page_output[id] = (True, True)

        elif A_context_score > B_context_score:
            page_output[id] = (True, False) 
        
        elif A_context_score < B_context_score: 
            page_output[id] = (False, True) 
    
    return page_output


"""
Main Part 
"""

# 1. Data Processing
gap_raw_data = load_raw_data()
gap_raw_data = gap_raw_data[1:]

try:
    with open('gap_data_dic.pickle', 'rb') as f :
        gap_data_dic = pickle.load(f) 

except FileNotFoundError:
    gap_data_dic = Preprocess(gap_raw_data)
    with open('gap_data_dic.pickle', 'wb') as f :
        pickle.dump(gap_data_dic, f, protocol=pickle.HIGHEST_PROTOCOL)

# 2. Snippet-Context 
try:
    with open('snippet_output.pickle', 'rb') as f1:
        snippet_output = pickle.load(f1)

except FileNotFoundError:
    snippet_output = Snippet_Context(gap_data_dic)
    with open('snippet_output.pickle', 'wb') as f1: 
        pickle.dump(snippet_output, f1, protocol=pickle.HIGHEST_PROTOCOL)

# print("------Snippet Results ------------")
# answer_snippet = 0
# for i in range(1, 2001):
#     if snippet_output[i] == (True, False) or snippet_output[i] == (False, True):
#         answer_snippet +=1 
# print(answer_snippet)

# 3. Page-Context 
try:
    with open('page_output.pickle', 'rb') as f2:
        page_output = pickle.load(f2)

except FileNotFoundError:
    page_output = Page_Context(gap_data_dic, snippet_output)
    with open('page_output.pickle', 'wb') as f2: 
        pickle.dump(page_output, f2, protocol=pickle.HIGHEST_PROTOCOL)

# print("------Page Results ------------")
# page_answer = 0
# for i in range(1, 2001):
#     if page_output[i] == (True, False) or page_output[i] == (False, True):
#         page_answer +=1 
# print(page_answer)

# 4. Data conversion to .tsv file. 
# snippet
with open("CS372_HW5_snippet_output_20160255.tsv", 'w', encoding='utf-8') as f: 
    tsv_writer = csv.writer(f, delimiter='\t')
    for id in range(1, 2001):
        (A_p, B_p) = snippet_output[id]
        v_id = 'test-'+str(id)
        tsv_writer.writerow([v_id, A_p, B_p])

# page
with open("CS372_HW5_page_output_20160255.tsv", 'w', encoding='utf-8') as f: 
    tsv_writer = csv.writer(f, delimiter='\t')
    for id in range(1, 2001):
        (A_p, B_p) = page_output[id]
        v_id = 'test-'+str(id)
        tsv_writer.writerow([v_id, A_p, B_p])














