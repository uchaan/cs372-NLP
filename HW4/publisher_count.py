import csv
from collections import defaultdict

verb_dic = defaultdict(int) 
pub_dic = defaultdict(int)

with open('medline.csv', 'r', encoding='utf-8') as f : 
    rdr = csv.reader(f) 
    for sentences in rdr : 
        _ ,annotate, _, publisher, _ = sentences 
        l = annotate.split(',')
        pub_dic[publisher]+=1
        verb_dic[l[1]]+=1

# print (dic)
print(pub_dic)
