import csv
from bert_serving.client import BertClient

pathout = './datasets/output_google_train.csv'
pathin = './datasets/google_news_vector_train.csv'
bc = BertClient()

table = []

with open(pathout, newline = '') as csvfile:
    date = []
    title = []
    vector = []
    data = csv.reader(csvfile)
    for row in data:
        newline = []
        newline.append(row[0])
        try:
            newline.append(row[1])
            a = bc.encode([row[1]])
            newline.extend(a[0])
        except:
            del newline[1]
            newline.extend([0 for i in range(1025)])
        table.append(newline)


with open(pathin, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table)
    
print('extraction finished!')

a = open(pathin, newline='')
b = csv.reader(a)
c = 0
for data in b:
    if c < 20:
        print('1:', len(data))
        print('2:', data[-3:])
    c+=1
