sentence = 'Clinical management aims to preserve spermatogenesis and prevent the increased risk of seminoma.'

annotation = 'Clinical management,prevent,the increased risk of seminoma'

title = 'The Undescended Testis: Clinical Management and Scientific Advances'

publisher = 'Semin Pediatr Surg'

year = '2016'

import csv 

f = open('medline.csv', 'a', encoding='utf-8')
wr = csv.writer(f) 
wr.writerow([sentence, annotation, title, publisher, year])
f.close()

