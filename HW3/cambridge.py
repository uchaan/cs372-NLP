import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup 
import ssl
import csv

# american = ['ɚ','ɝ','ɪr','ɛr','ʊr','ɔr','ɑr'] 
# WEBCRAWLING,
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'https://pubmed.ncbi.nlm.nih.gov/32474676/'
html = urllib.request.urlopen(url,context=ctx).read()
soup = BeautifulSoup(html, 'html.parser')

pron = soup.find_all("option", {"value": "abstract"})
print(pron)

# n = 1
# while(True): 
#     etymology = soup.find("span", {"class": "mw-headline", "id": "Etymology_"+str(n)})
#     if etymology==None:
#         break
#     if len(etymology.get_text().split()) == 1 :
#         break
#     n+=1
# n-=1
# print(n)

# pron = soup.find_all("span", {"class": "IPA"})
# count = 0
# for p in pron: 
#     p = p.get_text()
#     if p.count('/')==2:
#         no = 0
#         for letter in p :
#             if letter in american:
#                 no = 1
#         if no == 0: 
#             print(count+1, p) 
#             count +=1 

#     if count == n:
#         break

# definition = soup.find_all("ol")
# for i in range(n): 
#     print(i+1, definition[i].get_text().split('\n')[0])
