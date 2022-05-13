
# converts the simple label format to mAP format (https://github.com/Cartucho/mAP), each image has its own label file


import csv
import os

with open('train.txt', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        file_name = row[0]
        head, tail = os.path.split(file_name)
        base = os.path.splitext(tail)[0]
        print(base)
        new_file = 'ground-truth/' + base + '.txt'
        txt = row[5] + ' ' +  row[1] + ' ' +  row[2] + ' ' +  row[3] + ' ' +  row[4] + '\n'

        file1 = open(new_file,"w")
        file1.write(txt)
        file1.close() 
        

